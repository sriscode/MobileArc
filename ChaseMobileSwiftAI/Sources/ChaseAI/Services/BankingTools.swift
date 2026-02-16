// BankingTools.swift
// Tool Protocol implementations for Foundation Models
// The on-device LLM calls these when it needs real account data or wants to stage an action.
// READ-ONLY tools execute immediately. WRITE tools (StageTransferTool) create drafts only.

import Foundation
import FoundationModels

// MARK: - Get Account Balance (read-only)

struct GetAccountBalanceTool: Tool {
    let name        = "getAccountBalance"
    let description = "Fetch the current balance for a Chase account. Call this whenever the user asks about their balance — never guess the amount."

    @Generable
    struct Arguments {
        @Guide(description: "Which account to check: checking, savings, or credit")
        var accountType: ToolAccountType
    }

    @Generable
    enum ToolAccountType: String {
        case checking, savings, credit
    }

    func call(arguments: Arguments) async throws -> String {
        let cacheKey = "getAccountBalance:\(arguments.accountType)"

        // Check cache first
        if let cached = await ToolResponseCache.shared.get(cacheKey) {
            return cached    // ← returns immediately, no API call
        }
        
        let balance = try await AccountsAPI.shared.balance(type: arguments.accountType.rawValue)
        let str =  String("""
            \(arguments.accountType.rawValue.capitalized) account
            Current balance:   $\(String(format: "%.2f", balance.current))
            Available balance: $\(String(format: "%.2f", balance.available))
            Pending:           $\(String(format: "%.2f", balance.pending))
            Updated: \(balance.timestamp)
            """)
        
        await ToolResponseCache.shared.set(
            cacheKey,
            value: str,
            ttl: ToolResponseCache.TTL.accountBalance   // 24 hours
        )
        
        return str
    }
}

// MARK: - Get Transactions (read-only)

struct GetTransactionsTool: Tool {
    let name        = "getTransactions"
    let description = "Fetch recent transactions for spending analysis or finding specific purchases. Use when the user asks about spending, purchases, or transaction history."

    @Generable
    struct Arguments {
        @Guide(description: "How many days back to fetch (1–90)")
        var daysPast: Int

        @Guide(description: "Category to filter by: dining, groceries, transport, shopping, utilities, entertainment, healthcare, or 'all'")
        var category: String
    }

    func call(arguments: Arguments) async throws -> String {
        let cacheKey = "getTransactions: all)"

        if let cached = await ToolResponseCache.shared.get(cacheKey) {
            return cached
        }
        
        let safeDays = min(max(arguments.daysPast, 1), 90)
        let transactions = try await TransactionsAPI.shared.fetch(
            days: safeDays,
            category: arguments.category == "all" ? nil : arguments.category
        )

        // CRITICAL: raw card/account numbers never appear in tool output
        let lines = transactions.prefix(30).map { txn in
            "\(txn.date) | \(txn.merchant) | \(txn.formattedAmount) | \(txn.category)"
        }.joined(separator: "\n")

        let total = transactions.reduce(0.0) { $0 + $1.amount }

        let str =  String("""
            Last \(safeDays) days — \(transactions.count) transactions — Total: $\(String(format: "%.2f", total))

            \(lines)
            \(transactions.count > 30 ? "... and \(transactions.count - 30) more transactions" : "")
            """)
        
        await ToolResponseCache.shared.set(
            cacheKey,
            value: str,
            ttl: ToolResponseCache.TTL.transactions   // 24 hours
        )
        
        return str
    }
}

// MARK: - Stage Transfer (creates DRAFT only — never executes money movement)

struct StageTransferTool: Tool {
    let name        = "stageTransfer"
    let description = """
        Create a DRAFT transfer for the user to review and confirm.
        IMPORTANT: This does NOT move any money. It only creates a preview.
        The user must tap "Confirm Transfer" in the app to proceed.
        Always tell the user what you are about to stage before calling this tool.
        Hard limit: $10,000 per transfer. Decline amounts above this.
        """

    @Generable
    struct Arguments {
        @Guide(description: "Source account: 'checking' or 'savings'")
        var fromAccount: String

        @Guide(description: "Destination: 'checking', 'savings', or a Zelle recipient name/email")
        var toAccount: String

        @Guide(description: "Transfer amount in USD — must be between $0.01 and $10,000")
        var amount: Double

        @Guide(description: "Optional memo or note for the transfer. Leave empty if none.")
        var memo: String
    }

    func call(arguments: Arguments) async throws -> String {
        // Hard guardrail — cannot be bypassed by any prompt
        guard arguments.amount >= 0.01 else {
            return String("Error: Transfer amount must be at least $0.01.")
        }
        guard arguments.amount <= 10_000 else {
            return String("""
                Error: AI-initiated transfers are limited to $10,000.
                For larger amounts, please visit a Chase branch or call 1-800-935-9935.
                """)
        }

        let draft = try await TransferService.shared.createDraft(
            from: arguments.fromAccount,
            to:   arguments.toAccount,
            amount: arguments.amount,
            memo:   arguments.memo
        )

        // Post to main actor — triggers the SwiftUI confirmation sheet
        await MainActor.run {
            ApprovalCoordinator.shared.pendingDraft = draft
            ApprovalCoordinator.shared.showApproval = true
        }

        return String("""
            ✓ Transfer draft created and ready for review:
              From:   \(draft.fromAccountDisplay)
              To:     \(draft.toAccountDisplay)
              Amount: \(draft.formattedAmount)
              Memo:   \(draft.memo.isEmpty ? "none" : draft.memo)
              Draft ID: \(draft.id)

            A confirmation dialog has appeared in the app.
            The transfer will only proceed if the user taps Confirm.
            """)
    }
}

// MARK: - Get Savings Rates (read-only)

struct GetSavingsRatesTool: Tool {
    let name        = "getSavingsRates"
    let description = "Fetch current savings rates from Chase and the broader market. Use when giving savings or investment advice."

    @Generable
    struct Arguments {
        @Guide(description: "Type of rate: 'hysa', 'cd_6mo', 'cd_1yr', 'cd_2yr', or 'money_market'")
        var rateType: String
    }

    func call(arguments: Arguments) async throws -> String {
        let cacheKey = "getSavingsRates"

        if let cached = await ToolResponseCache.shared.get(cacheKey) {
            return cached
        }
        
        let rates = try await RatesAPI.shared.fetchSavingsRates(type: arguments.rateType)
//        let str =  String("""
//            \(arguments.rateType.uppercased()) rates (as of \(rates.date)):
//            Chase:          \(rates.chase)% APY
//            Best market:    \(rates.bestMarket)% APY
//            National avg:   \(rates.nationalAvg)% APY
//            """)
        
        let str =  String("""
            \(arguments.rateType.uppercased()) rates (as of \(rates.date)):
            Chase:          \(rates.chase)% APY
            Suggestion:    \(rates.action)
            """)
        
        await ToolResponseCache.shared.set(
            cacheKey,
            value: str,
            ttl: ToolResponseCache.TTL.savingsRates   // 24 hours
        )
        
        return str
    }
}

// MARK: - Get Credit Score (read-only)

struct GetCreditScoreTool: Tool {
    let name        = "getCreditScore"
    let description = "Fetch the user's current credit score and key factors from Chase Credit Journey."

    @Generable
    struct Arguments {
        @Guide(description: "Set true to include the individual score factors and their impact")
        var includeFactors: Bool
    }

    func call(arguments: Arguments) async throws -> String {
        let cacheKey = "getCreditScore"

        if let cached = await ToolResponseCache.shared.get(cacheKey) {
            return cached
        }
        let score = try await CreditJourneyAPI.shared.fetchScore()
        var output = """
            Credit score: \(score.score) (\(score.rating))
            Monthly change: \(score.change >= 0 ? "+" : "")\(score.change) pts
            Updated: \(score.date)
            """

        if arguments.includeFactors && !score.factors.isEmpty {
            output += "\n\nKey factors:\n"
            output += score.factors.map { "• \($0.name): \($0.impact)" }.joined(separator: "\n")
        }
        let str = String(output)
        
        await ToolResponseCache.shared.set(
            cacheKey,
            value: str,
            ttl: ToolResponseCache.TTL.creditScore   // 24 hours
        )
        
        return str
    }
}
