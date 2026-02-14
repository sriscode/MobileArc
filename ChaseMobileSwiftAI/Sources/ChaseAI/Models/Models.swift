// Models.swift
// All shared data models for Chase Agentic AI
// One file keeps imports simple across the project

import Foundation
import CoreLocation

// MARK: - Account

struct Account: Identifiable, Codable {
    let id:               String
    let type:             AccountType
    let displayName:      String
    let lastFourDigits:   String
    let balance:          Double
    let availableBalance: Double

    var maskedNumber: String { "••••\(lastFourDigits)" }
}

enum AccountType: String, Codable {
    case checking, savings, credit, investment
}

struct AccountBalance {
    let current:   Double
    let available: Double
    let pending:   Double
    let timestamp: String
}

// MARK: - Account Context
// Passed to every AI call — always sanitised before reaching any LLM

struct AccountContext {
    let userId:             String
    let accounts:           [Account]
    let recentTransactions: [Transaction]
    let creditScore:        Int?

    /// Safe summary — no card numbers, account numbers, SSNs, or PINs
    func toNonSensitiveSummary() -> String {
        let accountLines = accounts.map {
            "\($0.type.rawValue.capitalized) (\($0.maskedNumber)): $\(String(format: "%.2f", $0.balance))"
        }.joined(separator: " | ")

        let txnCount  = recentTransactions.count
        let totalSpent = recentTransactions.reduce(0.0) { $0 + $1.amount }

        return """
            Accounts: \(accountLines.isEmpty ? "none loaded" : accountLines)
            Recent activity: \(txnCount) transactions, $\(String(format: "%.2f", totalSpent)) total
            Credit score: \(creditScore.map(String.init) ?? "not available")
            """
    }

    static func current() -> AccountContext {
        AccountContext(
            userId:             "user_\(UUID().uuidString.prefix(8))",
            accounts:           [],
            recentTransactions: [],
            creditScore:        nil
        )
    }
}

// MARK: - Transaction

struct Transaction: Identifiable, Codable {
    let id:                   String
    let merchant:             String
    let amount:               Double
    let date:                 String
    let timestamp:            Date
    let category:             String
    let isCardNotPresent:     Bool
    let coordinate:           CLLocationCoordinate2D?
    let merchantCategoryCode: String

    var formattedAmount: String { "$\(String(format: "%.2f", amount))" }

    // Risk signal used by fraud detector
    var merchantCategoryRiskScore: Double {
        // High-risk MCCs: gambling, ATM, cash advance, money transfer
        let highRisk = ["7995", "6011", "6010", "6012", "4829"]
        return highRisk.contains(merchantCategoryCode) ? 0.85 : 0.20
    }

    init(id: String, merchant: String, amount: Double, date: String,
         timestamp: Date, category: String, isCardNotPresent: Bool = false,
         coordinate: CLLocationCoordinate2D? = nil,
         merchantCategoryCode: String = "5411") {
        self.id = id; self.merchant = merchant; self.amount = amount
        self.date = date; self.timestamp = timestamp; self.category = category
        self.isCardNotPresent = isCardNotPresent; self.coordinate = coordinate
        self.merchantCategoryCode = merchantCategoryCode
    }

    // Manual Codable for CLLocationCoordinate2D
    enum CodingKeys: String, CodingKey {
        case id, merchant, amount, date, timestamp, category,
             isCardNotPresent, merchantCategoryCode, latitude, longitude
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(String.self, forKey: .id)
        merchant = try c.decode(String.self, forKey: .merchant)
        amount = try c.decode(Double.self, forKey: .amount)
        date = try c.decode(String.self, forKey: .date)
        timestamp = try c.decode(Date.self, forKey: .timestamp)
        category = try c.decode(String.self, forKey: .category)
        isCardNotPresent = try c.decodeIfPresent(Bool.self, forKey: .isCardNotPresent) ?? false
        merchantCategoryCode = try c.decodeIfPresent(String.self, forKey: .merchantCategoryCode) ?? "5411"
        if let lat = try c.decodeIfPresent(Double.self, forKey: .latitude),
           let lon = try c.decodeIfPresent(Double.self, forKey: .longitude) {
            coordinate = CLLocationCoordinate2D(latitude: lat, longitude: lon)
        } else { coordinate = nil }
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(id, forKey: .id); try c.encode(merchant, forKey: .merchant)
        try c.encode(amount, forKey: .amount); try c.encode(date, forKey: .date)
        try c.encode(timestamp, forKey: .timestamp); try c.encode(category, forKey: .category)
        try c.encode(isCardNotPresent, forKey: .isCardNotPresent)
        try c.encode(merchantCategoryCode, forKey: .merchantCategoryCode)
        if let coord = coordinate {
            try c.encode(coord.latitude, forKey: .latitude)
            try c.encode(coord.longitude, forKey: .longitude)
        }
    }
}

// MARK: - Transfer

struct TransferDraft: Identifiable {
    let id:          String
    let fromAccount: String
    let toAccount:   String
    let amount:      Double
    let memo:        String
    let createdAt:   Date

    var fromAccountDisplay: String {
        "Chase \(fromAccount.capitalized)"
    }
    var toAccountDisplay: String {
        "Chase \(toAccount.capitalized)"
    }
    var formattedAmount: String {
        String(format: "$%.2f", amount)
    }
}

// MARK: - Rates & Credit

struct SavingsRates {
    let chase:       Double
    let bestMarket:  Double
    let nationalAvg: Double
    let date:        String
}

struct CreditScore {
    let score:   Int
    let rating:  String
    let change:  Int
    let date:    String
    let factors: [CreditFactor]
}

struct CreditFactor {
    let name:   String
    let impact: String
}

// MARK: - Chat

struct ChatMessage: Identifiable {
    let id        = UUID()
    let role:     MessageRole
    let content:  String
    let timestamp: Date
    var isStreaming: Bool = false

    enum MessageRole { case user, assistant, error }

    static func user(_ text: String)      -> ChatMessage { .init(role: .user,      content: text, timestamp: .now) }
    static func assistant(_ text: String) -> ChatMessage { .init(role: .assistant, content: text, timestamp: .now) }
    static func error(_ text: String)     -> ChatMessage { .init(role: .error,     content: text, timestamp: .now) }
}

// MARK: - Fraud Signal

struct FraudSignal {
    let transactionId:         String
    let merchant:              String
    let amount:                Double
    let anomalyScore:          Float
    let isSuspicious:          Bool
    let confidence:            Float
    let requiresImmediateReview: Bool
}
