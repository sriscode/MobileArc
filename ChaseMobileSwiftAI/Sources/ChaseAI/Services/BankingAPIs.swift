// BankingAPIs.swift
// Banking API clients: accounts, transactions, transfers, rates, credit score
// Development mode uses mock data. Replace each implementation with real Chase API calls.

import Foundation

// MARK: - Accounts API
@MainActor
class AccountsAPI {
    static let shared = AccountsAPI()
    private let baseURL = ProcessInfo.processInfo.environment["CHASE_API_BASE"]
                       ?? "https://api.chase.com/v1"

    func balance(type: String) async throws -> AccountBalance {
        // PRODUCTION: replace with real authenticated API call:
        // let (data, _) = try await URLSession.shared.data(for: authedRequest("/accounts/\(type)/balance"))
        // return try JSONDecoder().decode(AccountBalance.self, from: data)

        try await Task.sleep(nanoseconds: 200_000_000)   // simulate network
        switch type {
        case "checking":
            return AccountBalance(current: 4_821.50, available: 4_721.50, pending: 100.00, timestamp: "Just now")
        case "savings":
            return AccountBalance(current: 12_450.00, available: 12_450.00, pending: 0, timestamp: "Just now")
        case "credit":
            return AccountBalance(current: -1_240.00, available: 8_760.00, pending: -85.00, timestamp: "Just now")
        default:
            return AccountBalance(current: 0, available: 0, pending: 0, timestamp: "Unknown")
        }
    }

    func fetchAccounts() async throws -> [Account] {
        try await Task.sleep(nanoseconds: 300_000_000)
        return [
            Account(id: "chk_001", type: .checking, displayName: "Chase Total Checking",
                    lastFourDigits: "4821", balance: 4_821.50, availableBalance: 4_721.50),
            Account(id: "sav_001", type: .savings,  displayName: "Chase Savings",
                    lastFourDigits: "9034", balance: 12_450.00, availableBalance: 12_450.00),
            Account(id: "crd_001", type: .credit,   displayName: "Chase Sapphire Preferred",
                    lastFourDigits: "1337", balance: -1_240.00, availableBalance: 8_760.00),
        ]
    }
}

// MARK: - Transactions API

@MainActor
class TransactionsAPI {
    static let shared = TransactionsAPI()

    private let mockTransactions: [Transaction] = {
        let now = Date()
        return [
            Transaction(id: "t001", merchant: "Whole Foods Market",  amount: 84.32,  date: "Today",       timestamp: now,              category: "groceries"),
            Transaction(id: "t002", merchant: "Uber Eats",           amount: 31.50,  date: "Today",       timestamp: now - 3_600,      category: "dining"),
            Transaction(id: "t003", merchant: "Netflix",             amount: 15.49,  date: "Yesterday",   timestamp: now - 86_400,     category: "entertainment"),
            Transaction(id: "t004", merchant: "Shell Gas Station",   amount: 52.00,  date: "Yesterday",   timestamp: now - 90_000,     category: "transport"),
            Transaction(id: "t005", merchant: "Chipotle",            amount: 14.75,  date: "2 days ago",  timestamp: now - 172_800,    category: "dining"),
            Transaction(id: "t006", merchant: "Amazon.com",          amount: 67.99,  date: "3 days ago",  timestamp: now - 259_200,    category: "shopping",   isCardNotPresent: true),
            Transaction(id: "t007", merchant: "Starbucks",           amount: 8.45,   date: "3 days ago",  timestamp: now - 270_000,    category: "dining"),
            Transaction(id: "t008", merchant: "Planet Fitness",      amount: 24.99,  date: "5 days ago",  timestamp: now - 432_000,    category: "utilities"),
            Transaction(id: "t009", merchant: "Trader Joe's",        amount: 92.10,  date: "6 days ago",  timestamp: now - 518_400,    category: "groceries"),
            Transaction(id: "t010", merchant: "United Airlines",     amount: 389.00, date: "1 week ago",  timestamp: now - 604_800,    category: "transport",  isCardNotPresent: true),
            Transaction(id: "t011", merchant: "Costco Wholesale",    amount: 187.42, date: "1 week ago",  timestamp: now - 691_200,    category: "groceries"),
            Transaction(id: "t012", merchant: "Apple.com/Bill",      amount: 9.99,   date: "10 days ago", timestamp: now - 864_000,    category: "entertainment", isCardNotPresent: true),
            Transaction(id: "t013", merchant: "CVS Pharmacy",        amount: 23.40,  date: "12 days ago", timestamp: now - 1_036_800,  category: "shopping"),
            Transaction(id: "t014", merchant: "DoorDash",            amount: 42.15,  date: "2 weeks ago", timestamp: now - 1_296_000,  category: "dining"),
            Transaction(id: "t015", merchant: "Spotify",             amount: 11.99,  date: "2 weeks ago", timestamp: now - 1_382_400,  category: "entertainment", isCardNotPresent: true),
        ]
    }()

    func fetch(days: Int, category: String?) async throws -> [Transaction] {
        try await Task.sleep(nanoseconds: 150_000_000)
        let cutoff = Date().addingTimeInterval(-Double(days * 86_400))
        return mockTransactions
            .filter { $0.timestamp > cutoff }
            .filter { category == nil || $0.category == category }
    }
}

// MARK: - Transfer Service
@MainActor
class TransferService {
    static let shared = TransferService()

    func createDraft(from: String, to: String, amount: Double, memo: String) async throws -> TransferDraft {
        try await Task.sleep(nanoseconds: 100_000_000)
        return TransferDraft(
            id:          "draft_\(UUID().uuidString.prefix(8))",
            fromAccount: from,
            toAccount:   to,
            amount:      amount,
            memo:        memo,
            createdAt:   Date()
        )
    }

    func executeTransfer(_ draft: TransferDraft, confirmationToken: String) async throws {
        try await Task.sleep(nanoseconds: 500_000_000)
        // PRODUCTION: POST to /v1/transfers with signed confirmation token
        AuditLogger.shared.log(event: "transfer_executed", metadata: [
            "draft_id":   draft.id,
            "amount":     String(draft.amount),
            "from_last4": String(draft.fromAccount.suffix(4)),
            "to_last4":   String(draft.toAccount.suffix(4)),
        ])
    }
}

// MARK: - Rates API
@MainActor
class RatesAPI {
    static let shared = RatesAPI()

    func fetchSavingsRates(type: String) async throws -> SavingsRates {
        try await Task.sleep(nanoseconds: 200_000_000)
        switch type {
        case "hysa":    return SavingsRates(chase: 0.01,  bestMarket: 5.25, nationalAvg: 0.58, date: "Today", action: "Open High Yield returns Account")
        case "cd_6mo":  return SavingsRates(chase: 4.25,  bestMarket: 5.40, nationalAvg: 1.82, date: "Today", action: "Open CD for 6 months")
        case "cd_1yr":  return SavingsRates(chase: 4.50,  bestMarket: 5.35, nationalAvg: 1.89, date: "Today", action: "Open CD for 1 year")
        case "cd_2yr":  return SavingsRates(chase: 4.30,  bestMarket: 5.10, nationalAvg: 1.75, date: "Today", action: "Open CD for 2 years")
        default:        return SavingsRates(chase: 0.01,  bestMarket: 5.00, nationalAvg: 0.50, date: "Today", action: "Save using AutoSave Feature")
        }
    }
}

// MARK: - Credit Journey API
@MainActor
class CreditJourneyAPI {
    static let shared = CreditJourneyAPI()

    func fetchScore() async throws -> CreditScore {
        try await Task.sleep(nanoseconds: 300_000_000)
        return CreditScore(
            score:  742,
            rating: "Very Good",
            change: +8,
            date:   "Updated today",
            factors: [
                CreditFactor(name: "Payment History",    impact: "Excellent — 100% on-time"),
                CreditFactor(name: "Credit Utilization", impact: "Good — 18% utilization"),
                CreditFactor(name: "Credit Age",         impact: "Good — 7yr 4mo average"),
                CreditFactor(name: "Hard Inquiries",     impact: "1 inquiry in last 12 months"),
                CreditFactor(name: "Credit Mix",         impact: "Good — cards + auto loan"),
            ]
        )
    }
}

// MARK: - Approval Coordinator (SwiftUI bridge for transfer confirmation sheet)

@Observable @MainActor
class ApprovalCoordinator {
    static let shared = ApprovalCoordinator()
    var pendingDraft: TransferDraft?
    var showApproval = false
}

// MARK: - Audit Logger
@MainActor
class AuditLogger {
    static let shared = AuditLogger()

    func log(event: String, metadata: [String: String]) {
        // PRODUCTION: Write to AWS QLDB (immutable append-only ledger)
        // Required for PCI-DSS and FFIEC compliance
        var entry = metadata
        entry["event"]     = event
        entry["timestamp"] = ISO8601DateFormatter().string(from: Date())
        print("AUDIT: \(entry)")    // swap for QLDB write in production
    }
}
