// CloudAgentService.swift
// Swift client for the Python FastAPI + LangGraph cloud agent.
// Called for: money transfers, investment queries, real-time market data.
// Every financial execution requires a human approval token — no exceptions.

import Foundation

// MARK: - Request / Response Types

private struct CloudQueryRequest: Encodable {
    let query:             String
    let intent:            String
    let contextSummary:    String   // sanitised — no raw PII
    let approvedDraftIds:  [String]
    let sessionId:         String
}

private struct CloudQueryResponse: Decodable {
    let text:      String
    let actions:   [RemoteAction]?
    let sessionId: String
}

private struct RemoteAction: Decodable {
    let type:    String
    let payload: [String: String]
}

private struct ExecuteTransferRequest: Encodable {
    let draftId:           String
    let confirmationToken: String
}

// MARK: - Cloud Agent Service

class CloudAgentService {

    private let baseURL:  String
    private var sessionId = UUID().uuidString
    private var approvedDraftIds: [String] = []

    init() {
//        self.baseURL = ProcessInfo.processInfo.environment["CLOUD_AGENT_URL"]
//                    ?? "https://your-cloud-agent.com"
        self.baseURL = "http://localhost:8000"
    }

    // MARK: - Process Query

    nonisolated(nonsending) func process(
        query: String,
        intent: UserIntent,
        context: AccountContext
    ) async throws -> AgentResponse {
        let req = CloudQueryRequest(
            query:            query,
            intent:           intent.type.rawValue,
            contextSummary:   context.toNonSensitiveSummary(),
            approvedDraftIds: approvedDraftIds,
            sessionId:        sessionId
        )

        let resp: CloudQueryResponse = try await post(endpoint: "/agent/query", body: req)

        // Process any structured actions the cloud agent signals back
        for action in resp.actions ?? [] {
            if action.type == "transfer_staged",
               let draftId  = action.payload["draft_id"],
               let fromAcct = action.payload["from_account"],
               let toAcct   = action.payload["to_account"],
               let amtStr   = action.payload["amount"],
               let amount   = Double(amtStr) {
                let draft = TransferDraft(
                    id: draftId, fromAccount: fromAcct, toAccount: toAcct,
                    amount: amount, memo: action.payload["memo"] ?? "",
                    createdAt: Date()
                )
                return .transferStaged(draft)
            }

            if action.type == "fraud_alert",
               let txnId    = action.payload["transaction_id"],
               let merchant = action.payload["merchant"],
               let amtStr   = action.payload["amount"],
               let amount   = Double(amtStr) {
                return .fraudAlert(FraudSignal(
                    transactionId: txnId, merchant: merchant, amount: amount,
                    anomalyScore: -0.5, isSuspicious: true, confidence: 0.95,
                    requiresImmediateReview: true
                ))
            }
        }

        return .text(resp.text)
    }

    // MARK: - Execute Approved Transfer

    nonisolated(nonsending) func executeTransfer(_ draft: TransferDraft, confirmationToken: String) async throws {
        let req = ExecuteTransferRequest(
            draftId: draft.id,
            confirmationToken: confirmationToken
        )
        let _: CloudQueryResponse = try await post(endpoint: "/agent/transfer/execute", body: req)

        await AuditLogger.shared.log(event: "transfer_executed_confirmed", metadata: [
            "draft_id": draft.id,
            "amount":   String(draft.amount)
        ])
    }

    // MARK: - Spending Analysis Fallback (returns nil — use Foundation Models instead)

    nonisolated(nonsending) func analyzeSpending(transactions: [Transaction]) async throws -> SpendingReport? {
        return nil   // Foundation Models handles this on-device; cloud isn't needed
    }

    // MARK: - Approve Draft (called when user confirms in UI)

    func recordApproval(draftId: String) {
        approvedDraftIds.append(draftId)
    }

    // MARK: - HTTP Client

    nonisolated(nonsending) private func post<Req: Encodable, Resp: Decodable>(
        endpoint: String,
        body: Req
    ) async throws -> Resp {
        guard let url = URL(string: "\(baseURL)\(endpoint)") else {
            throw CloudError.invalidURL
        }

        var request        = URLRequest(url: url)
        request.httpMethod = "POST"
        request.httpBody   = try JSONEncoder().encode(body)
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 30

        // Attach biometric-backed auth token
        let security = AppSecurityService()
        if let token = try? await security.authenticateWithBiometrics() {
            request.setValue(
                "Bearer \(token.rawValue.base64EncodedString())",
                forHTTPHeaderField: "Authorization"
            )
        }

        let (data, response) = try await URLSession.shared.data(for: request)

        guard let http = response as? HTTPURLResponse else {
            throw CloudError.invalidResponse
        }
        guard http.statusCode == 200 else {
            throw CloudError.httpError(http.statusCode)
        }

        return try JSONDecoder().decode(Resp.self, from: data)
    }

    func resetSession() {
        sessionId        = UUID().uuidString
        approvedDraftIds = []
    }
}

// MARK: - Errors

enum CloudError: LocalizedError {
    case invalidURL
    case invalidResponse
    case httpError(Int)

    var errorDescription: String? {
        switch self {
        case .invalidURL:        return "Invalid server endpoint"
        case .invalidResponse:   return "Unexpected server response"
        case .httpError(let c):  return "Server error (\(c))"
        }
    }
}

