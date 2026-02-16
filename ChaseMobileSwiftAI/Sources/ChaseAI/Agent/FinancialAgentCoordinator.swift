// FinancialAgentCoordinator.swift
// The single entry point for ALL AI features in the app.
// iOS 26+ only: Foundation Models (on-device LLM) + Cloud (financial execution)
// Core ML classifiers always run regardless: intent, fraud, embeddings.

import Foundation
import FoundationModels




// MARK: - Backend Enum

enum AIBackend {
    case foundationModels   // iOS 26 + Apple Intelligence active (primary path)
    case cloud              // Complex tasks: transfers, investments, real-time data
    case unavailable        // iOS 26 device but Apple Intelligence disabled in Settings
}

// MARK: - Response Types

enum AgentResponse {
    case text(String)
    case structuredReport(SpendingReport)
    case insights([FinancialInsight])
    case fraudAlert(FraudSignal)
    case transferStaged(TransferDraft)
    case error(String)
}

// MARK: - Coordinator

@MainActor @Observable
public final class FinancialAgentCoordinator {

    // Services
    private(set) var activeBackend: AIBackend = .unavailable
    private(set) var isReady = false

    private let cloudAgent = CloudAgentActor()


    // Actor wrapper to isolate cloud service off the main actor and avoid sending non-Sendable instances across actors
    actor CloudAgentActor {
        private let service = CloudAgentService()

        func process(query: String, intent: UserIntent, context: AccountContext) async throws -> AgentResponse {
            return try await service.process(query: query, intent: intent, context: context)
        }

        func executeTransfer(_ draft: TransferDraft, confirmationToken: String) async throws {
            try await service.executeTransfer(draft, confirmationToken: confirmationToken)
        }
    }

    private var foundationService: FoundationModelsService?

    // Always-on Core ML services (independent of LLM backend)
    let intentClassifier = IntentClassifier()
    let fraudDetector    = FraudDetector()
//    let vectorStore      = LocalVectorStore()

    // Transfer approval state — observed by SwiftUI
    var pendingTransfer: TransferDraft?
    var showTransferApproval = false

    // MARK: - Init

    func initialize() async {
        activeBackend = resolveBackend()

        if activeBackend == .foundationModels {
            let svc = FoundationModelsService()
            await svc.prewarm()                     // reduces first-query latency
            self.foundationService = svc
        }

        // Warm up Core ML vector store in background — doesn't block UI
//        let store = self.vectorStore
//        Task(priority: .background) {
//            try? await store.initialize()
//        }

        isReady = true
    }

    // MARK: - Main Query Dispatch

    func process(query: String, context: AccountContext) async throws -> AgentResponse {
        guard isReady else { throw AgentError.notInitialized }

        // Capture references locally to avoid implicitly sending `self` across awaits
        let intentClassifier = self.intentClassifier
        let fraudDetector = self.fraudDetector
        let cloudAgent = self.cloudAgent
        let activeBackend = self.activeBackend
        let foundationService = self.foundationService

        // 1. Intent classification via Core ML — always fast (~5ms), routes the query
        var intent = (try? intentClassifier.classify(query))
            ?? UserIntent(type: .general, confidence: 0.5)
        
        // If classifier isn't confident, fall back to keywords
        if intent.confidence < 0.75 {
            intent = classifyWithKeywords(query)
        }

        // 2. Fraud check runs in parallel — scans latest transactions
        async let fraudCheck = fraudDetector.scanLatest(context.recentTransactions)

        // 3. Money movement / real-time data / investments → always cloud
        if intent.requiresCloudExecution {
            _ = await fraudCheck    // ensure fraud scan completes before cloud call
            return try await cloudAgent.process(query: query, intent: intent, context: context)
        }

        // 4. Surface any fraud alert before responding to conversational query
        if let signal = await fraudCheck {
            return .fraudAlert(signal)
        }

        // 5. Foundation Models handles all remaining conversational tasks
        guard activeBackend == .foundationModels, let fm = foundationService else {
            throw AgentError.foundationModelsUnavailable
        }
        return try await fm.respond(query: query, intent: intent, context: context)
    }

    // Spending analysis — Foundation Models streaming snapshot output (partial snapshots)
    func analyzeSpending(transactions: [Transaction]) async -> AsyncThrowingStream<SpendingReport.PartiallyGenerated, Error> {
        guard let fm = foundationService else {
            return AsyncThrowingStream { $0.finish(throwing: AgentError.foundationModelsUnavailable) }
        }
        return await fm.analyzeSpendingStream(transactions: transactions)
    }
    
    // MARK: - Hardcoded classification based on keywords
    private func classifyWithKeywords(_ query: String) -> UserIntent {
        let t = query.lowercased()
        if t.contains("spend") || t.contains("analys") || t.contains("transaction") || t.contains("bought") {
            return UserIntent(type: .spendingAnalysis, confidence: 0.85)
        }
        if t.contains("transfer") || t.contains("send") || t.contains("move") {
            return UserIntent(type: .transferRequest, confidence: 0.85)
        }
        if t.contains("invest") || t.contains("portfolio") || t.contains("stock") {
            return UserIntent(type: .investmentQuery, confidence: 0.85)
        }
        if t.contains("balance") || t.contains("how much") {
            return UserIntent(type: .balanceQuery, confidence: 0.85)
        }
        return UserIntent(type: .general, confidence: 0.5)
    }

    // Generate proactive insights for Home screen
    func generateInsights(context: AccountContext) async throws -> [FinancialInsight] {
        guard let fm = foundationService else { throw AgentError.foundationModelsUnavailable }
        return try await fm.generateInsights(context: context)
    }

    // MARK: - Transfer Approval Flow

    @MainActor func stageTransfer(_ draft: TransferDraft) async {
        pendingTransfer      = draft
        showTransferApproval = true
    }

    @MainActor func executeApprovedTransfer(_ draft: TransferDraft, confirmationToken: String) async throws {
        try await cloudAgent.executeTransfer(draft, confirmationToken: confirmationToken)
        pendingTransfer      = nil
        showTransferApproval = false
    }

    @MainActor func cancelTransfer() {
        pendingTransfer      = nil
        showTransferApproval = false
    }

    // MARK: - Chat Session Management

    func resetChatSession() {
        resetCache()
        Task { [foundationService] in
            await foundationService?.resetChatHistory()
        }
    }
    
    func resetAnalysisSession() {
        Task { [foundationService] in
            await foundationService?.resetAnalysisHistory()
        }
    }
    
    func resetCache() {
        Task { await ToolResponseCache.shared.invalidateAll() }
    }

    // MARK: - Backend Resolution

    private func resolveBackend() -> AIBackend {
        // Deployment target is iOS 26 — this guard is a safety net, never triggers at runtime
        guard #available(iOS 26, *) else {
            fatalError("ChaseAI requires iOS 26 or later")
        }

        if SystemLanguageModel.default.isAvailable {
            print("✅ Foundation Models active — on-device AI enabled")
            return .foundationModels
        }

        // Device runs iOS 26 but Apple Intelligence is:
        //   • Disabled in Settings → Apple Intelligence & Siri
        //   • Not yet downloaded (freshly enabled, model still downloading)
        //   • Unsupported locale/region for Apple Intelligence
        print("⚠️ Foundation Models unavailable — check Apple Intelligence in Settings")
        return .unavailable
    }
}

// MARK: - Errors

enum AgentError: LocalizedError {
    case notInitialized
    case foundationModelsUnavailable
    case cloudUnavailable
    case guardrailViolation(String)

    var errorDescription: String? {
        switch self {
        case .notInitialized:
            return "AI agent not yet initialized. Please wait."
        case .foundationModelsUnavailable:
            return "Apple Intelligence is not available. Enable it in Settings → Apple Intelligence & Siri."
        case .cloudUnavailable:
            return "An internet connection is required for this action."
        case .guardrailViolation(let reason):
            return "Action not permitted: \(reason)"
        }
    }
}

