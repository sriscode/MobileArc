// Views.swift
// Complete SwiftUI interface for Chase Agentic AI
// All screens: Home, Chat, Spending, Investments, Profile, Approval Sheet,
//              Launch, Biometric Auth, Apple Intelligence Required

import SwiftUI
import UIKit

// ═══════════════════════════════════════════════════════════════
// MARK: - Main Tab View
// ═══════════════════════════════════════════════════════════════

public struct MainTabView: View {
    @Environment(FinancialAgentCoordinator.self) var coordinator

    public var body: some View {
        @Bindable var coord = coordinator

        TabView {
            HomeView()
                .tabItem { Label("Home",     systemImage: "house.fill") }

            ChatView()
                .tabItem { Label("AI Agent", systemImage: "sparkles") }

            SpendingView()
                .tabItem { Label("Spending", systemImage: "chart.bar.fill") }

            InvestmentsView()
                .tabItem { Label("Invest",   systemImage: "chart.line.uptrend.xyaxis") }

            ProfileView()
                .tabItem { Label("Profile",  systemImage: "person.fill") }
        }
        .accentColor(.blue)
        .sheet(isPresented: $coord.showTransferApproval) {
            if let draft = coord.pendingTransfer {
                TransferApprovalSheet(draft: draft)
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - Home View
// ═══════════════════════════════════════════════════════════════

struct HomeView: View {
    @Environment(FinancialAgentCoordinator.self) var coordinator
    @State private var accounts: [Account]        = []
    @State private var insights: [FinancialInsight] = []
    @State private var isLoadingInsights = false
    @State private var loadError: String?

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 20) {

                    // AI status banner
                    AIGreeterCard(backend: coordinator.activeBackend)
                        .padding(.horizontal)

                    // Account cards (horizontal scroll)
                    if !accounts.isEmpty {
                        ScrollView(.horizontal, showsIndicators: false) {
                            HStack(spacing: 12) {
                                ForEach(accounts) { AccountCard(account: $0) }
                            }
                            .padding(.horizontal)
                        }
                    }

                    // Quick action grid
                    QuickActionsGrid()
                        .padding(.horizontal)

                    // AI-generated insights
                    VStack(alignment: .leading, spacing: 12) {
                        HStack {
                            Text("AI Insights")
                                .font(.headline)
                            Spacer()
                            if isLoadingInsights {
                                ProgressView().scaleEffect(0.8)
                            }
                        }
                        .padding(.horizontal)

                        if let err = loadError {
                            Text(err)
                                .font(.caption)
                                .foregroundColor(.secondary)
                                .padding(.horizontal)
                        } else {
                            ForEach(insights.indices, id: \.self) { i in
                                InsightCard(insight: insights[i])
                                    .padding(.horizontal)
                            }
                        }
                    }
                }
                .padding(.top)
            }
            .navigationTitle("Chase AI")
            .navigationBarTitleDisplayMode(.large)
        }
        .task { await loadData() }
    }

    private func loadData() async {
        accounts = (try? await AccountsAPI.shared.fetchAccounts()) ?? []
        isLoadingInsights = true
        do {
            insights = try await coordinator.generateInsights(context: .current())
        } catch {
            loadError = "Insights unavailable: \(error.localizedDescription)"
        }
        isLoadingInsights = false
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - AI Greeter Card
// ═══════════════════════════════════════════════════════════════

struct AIGreeterCard: View {
    let backend: AIBackend

    private var isActive: Bool { backend == .foundationModels }

    var body: some View {
        HStack(spacing: 14) {
            ZStack {
                Circle()
                    .fill(LinearGradient(
                        colors: isActive ? [.blue, .cyan] : [.gray, .gray.opacity(0.6)],
                        startPoint: .topLeading, endPoint: .bottomTrailing))
                    .frame(width: 44, height: 44)
                Text("✦")
                    .foregroundColor(.white)
                    .font(.system(size: 18, weight: .bold))
            }

            VStack(alignment: .leading, spacing: 3) {
                Text(isActive ? "Chase AI is active" : "Chase AI — limited")
                    .font(.headline)
                HStack(spacing: 5) {
                    Circle()
                        .fill(isActive ? Color.green : Color.orange)
                        .frame(width: 7, height: 7)
                    Text(backendLabel)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            Spacer()
        }
        .padding()
        .background(.regularMaterial)
        .cornerRadius(16)
    }

    var backendLabel: String {
        switch backend {
        case .foundationModels: return "Apple Foundation Models · On-device"
        case .cloud:            return "Cloud AI · Secure connection"
        case .unavailable:      return "Enable Apple Intelligence in Settings"
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - Account Card
// ═══════════════════════════════════════════════════════════════

struct AccountCard: View {
    let account: Account

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text(account.type.rawValue.capitalized)
                    .font(.caption)
                    .fontWeight(.semibold)
                    .foregroundColor(.secondary)
                    .textCase(.uppercase)
                Spacer()
                Image(systemName: accountIcon)
                    .foregroundColor(.blue)
            }

            Text(account.maskedNumber)
                .font(.system(.subheadline, design: .monospaced))
                .foregroundColor(.secondary)

            Text("$\(String(format: "%.2f", account.balance))")
                .font(.system(.title2, design: .rounded))
                .fontWeight(.bold)
                .foregroundColor(account.balance < 0 ? .red : .primary)
        }
        .padding()
        .frame(width: 190)
        .background(.regularMaterial)
        .cornerRadius(16)
    }

    var accountIcon: String {
        switch account.type {
        case .checking:   return "dollarsign.circle.fill"
        case .savings:    return "building.columns.fill"
        case .credit:     return "creditcard.fill"
        case .investment: return "chart.line.uptrend.xyaxis"
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - Quick Actions Grid
// ═══════════════════════════════════════════════════════════════

struct QuickActionsGrid: View {
    private let actions: [(String, String, Color)] = [
        ("Send",      "arrow.up.circle.fill",            .blue),
        ("Deposit",   "camera.fill",                     .green),
        ("Pay Bills", "doc.text.fill",                   .orange),
        ("ATM",       "mappin.circle.fill",               .red),
        ("Budget",    "chart.pie.fill",                  .purple),
        ("Rewards",   "gift.fill",                       .pink),
        ("Travel",    "airplane.circle.fill",             .cyan),
        ("Invest",    "chart.line.uptrend.xyaxis.circle.fill", .indigo),
    ]
    private let columns = Array(repeating: GridItem(.flexible()), count: 4)

    var body: some View {
        LazyVGrid(columns: columns, spacing: 16) {
            ForEach(actions, id: \.0) { action in
                VStack(spacing: 8) {
                    Image(systemName: action.1)
                        .font(.system(size: 26))
                        .foregroundColor(action.2)
                        .frame(width: 56, height: 56)
                        .background(action.2.opacity(0.12))
                        .cornerRadius(16)
                    Text(action.0)
                        .font(.caption2)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - Insight Card
// ═══════════════════════════════════════════════════════════════

struct InsightCard: View {
    let insight: FinancialInsight

    var body: some View {
        HStack(alignment: .top, spacing: 14) {
            Text(categoryEmoji)
                .font(.title2)

            VStack(alignment: .leading, spacing: 6) {
                Text(insight.headline)
                    .font(.subheadline).fontWeight(.semibold)

                Text(insight.detail)
                    .font(.caption).foregroundColor(.secondary)
                    .lineLimit(3)

                HStack(spacing: 8) {
                    if insight.dollarImpact > 0 {
                        Label("~$\(String(format: "%.0f", insight.dollarImpact))/yr", systemImage: "arrow.up.right")
                            .font(.caption2).fontWeight(.medium)
                            .foregroundColor(.green)
                            .padding(.horizontal, 8).padding(.vertical, 3)
                            .background(.green.opacity(0.12))
                            .cornerRadius(6)
                    }
                    Text(insight.action)
                        .font(.caption2)
                        .foregroundColor(.blue)
                        .lineLimit(1)
                }
            }
        }
        .padding()
        .background(.regularMaterial)
        .cornerRadius(14)
    }

    var categoryEmoji: String {
        switch insight.category {
        case .savings:    return "💰"
        case .fraud:      return "🚨"
        case .spending:   return "📊"
        case .investment: return "📈"
        case .credit:     return "💳"
        case .general:    return "✨"
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - Chat View
// ═══════════════════════════════════════════════════════════════

struct ChatView: View {
    @Environment(FinancialAgentCoordinator.self) var coordinator
    @State private var messages:   [ChatMessage] = []
    @State private var inputText   = ""
    @State private var isLoading   = false
    @FocusState private var focused: Bool

    private let suggestions = [
        "What's my balance?",
        "Analyse my spending",
        "Find savings opportunities",
        "Show recent transactions",
        "Help me dispute a charge",
        "Transfer $200 to savings"
    ]

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {

                // Message list
                ScrollViewReader { proxy in
                    ScrollView {
                        VStack(spacing: 12) {
                            if messages.isEmpty {
                                WelcomeView(suggestions: suggestions) { s in
                                    Task { await send(s) }
                                }
                                .padding()
                            }
                            ForEach(messages) { msg in
                                MessageBubble(message: msg).id(msg.id)
                            }
                            if isLoading {
                                TypingIndicator().id("typing")
                            }
                        }
                        .padding(.vertical)
                    }
                    .onChange(of: messages.count) { _, _ in
                        withAnimation { proxy.scrollTo(messages.last?.id, anchor: .bottom) }
                    }
                    .onChange(of: isLoading) { _, loading in
                        if loading { withAnimation { proxy.scrollTo("typing", anchor: .bottom) } }
                    }
                }

                Divider()

                // Suggestion pills (shown after first message)
                if !messages.isEmpty && !isLoading {
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 8) {
                            ForEach(suggestions, id: \.self) { s in
                                Button(s) { Task { await send(s) } }
                                    .font(.caption)
                                    .padding(.horizontal, 12).padding(.vertical, 7)
                                    .background(.blue.opacity(0.10))
                                    .foregroundColor(.blue)
                                    .cornerRadius(20)
                            }
                        }
                        .padding(.horizontal).padding(.vertical, 8)
                    }
                }

                // Input bar
                HStack(spacing: 10) {
                    TextField("Ask Chase AI…", text: $inputText, axis: .vertical)
                        .textFieldStyle(.plain)
                        .padding(.horizontal, 16).padding(.vertical, 10)
                        .background(Color(.secondarySystemBackground))
                        .cornerRadius(24)
                        .focused($focused)
                        .lineLimit(4)
                        .submitLabel(.send)
                        .onSubmit { Task { await send(inputText) } }

                    Button { Task { await send(inputText) } } label: {
                        Image(systemName: "arrow.up.circle.fill")
                            .font(.system(size: 34))
                            .foregroundColor(inputText.isEmpty || isLoading ? .gray : .blue)
                    }
                    .disabled(inputText.isEmpty || isLoading)
                }
                .padding(.horizontal).padding(.vertical, 10)
                .background(Color(.systemBackground))
            }
            .navigationTitle("Chase AI")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Clear") {
                        messages = []
                        coordinator.resetChatSession()
                    }
                    .font(.caption)
                    .disabled(messages.isEmpty)
                }
            }
        }
    }

    private func send(_ text: String) async {
        let query = text.trimmingCharacters(in: .whitespaces)
        guard !query.isEmpty else { return }
        inputText = ""; focused = false

        messages.append(.user(query))
        isLoading = true

        do {
            let context  = AccountContext.current()
            let response = try await coordinator.process(query: query, context: context)
            switch response {
            case .text(let t):
                messages.append(.assistant(t))
            case .fraudAlert(let signal):
                messages.append(.assistant(
                    "🚨 Suspicious transaction detected!\n\n**\(signal.merchant)** — \(signal.formattedAmount)\n\nThis transaction looks unusual. Tap 'Dispute' in your activity to file a claim."
                ))
            case .transferStaged(let draft):
                messages.append(.assistant(
                    "I've prepared a transfer of \(draft.formattedAmount) for your review. Please check the confirmation dialog that just appeared."
                ))
            default:
                messages.append(.assistant("Done — let me know if you need anything else."))
            }
        } catch {
            messages.append(.error("⚠️ \(error.localizedDescription)"))
        }

        isLoading = false
    }
}

// MARK: - Message Bubble

struct MessageBubble: View {
    let message: ChatMessage

    var body: some View {
        HStack(alignment: .bottom, spacing: 8) {
            if message.role == .user { Spacer(minLength: 60) }

            Text(message.content)
                .font(.body)
                .padding(.horizontal, 16).padding(.vertical, 12)
                .background(bubbleColor)
                .foregroundColor(textColor)
                .cornerRadius(20)
                .cornerRadius(message.role == .user ? 4 : 20,
                              corners: message.role == .user ? .bottomRight : .bottomLeft)

            if message.role != .user { Spacer(minLength: 60) }
        }
        .padding(.horizontal)
    }

    var bubbleColor: Color {
        switch message.role {
        case .user:      return .blue
        case .assistant: return Color(.secondarySystemBackground)
        case .error:     return .red.opacity(0.12)
        }
    }
    var textColor: Color { message.role == .user ? .white : .primary }
}

// MARK: - Welcome View

struct WelcomeView: View {
    let suggestions: [String]
    let onSelect: (String) -> Void

    var body: some View {
        VStack(spacing: 24) {
            Image(systemName: "sparkles")
                .font(.system(size: 50))
                .foregroundStyle(LinearGradient(
                    colors: [.blue, .cyan], startPoint: .top, endPoint: .bottom))

            VStack(spacing: 8) {
                Text("Chase AI Agent")
                    .font(.title2).fontWeight(.bold)
                Text("I have secure access to your accounts and can help with balances, spending, transfers, and more.")
                    .font(.subheadline).foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
            }

            VStack(spacing: 10) {
                ForEach(suggestions, id: \.self) { s in
                    Button { onSelect(s) } label: {
                        HStack {
                            Text(s).font(.subheadline)
                            Spacer()
                            Image(systemName: "chevron.right").font(.caption).foregroundColor(.blue)
                        }
                        .padding()
                        .background(Color(.secondarySystemBackground))
                        .cornerRadius(12)
                        .foregroundColor(.primary)
                    }
                }
            }
        }
    }
}

// MARK: - Typing Indicator

struct TypingIndicator: View {
    @State private var animating = false

    var body: some View {
        HStack(spacing: 5) {
            ForEach(0..<3, id: \.self) { i in
                Circle()
                    .fill(Color.secondary)
                    .frame(width: 8, height: 8)
                    .scaleEffect(animating ? 1.3 : 0.8)
                    .animation(
                        .easeInOut(duration: 0.5)
                        .repeatForever()
                        .delay(Double(i) * 0.18),
                        value: animating
                    )
            }
        }
        .padding(.horizontal, 16).padding(.vertical, 12)
        .background(Color(.secondarySystemBackground))
        .cornerRadius(20)
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal)
        .onAppear { animating = true }
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - Spending View
// ═══════════════════════════════════════════════════════════════

struct SpendingView: View {
    @Environment(FinancialAgentCoordinator.self) var coordinator
    @State private var report:      SpendingReport.PartiallyGenerated?
    @State private var isAnalysing  = false
    @State private var transactions: [Transaction] = []

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 20) {
                    if isAnalysing {
                        VStack(spacing: 16) {
                            ProgressView()
                                .scaleEffect(1.4)
                            Text("AI is analysing your spending…")
                                .font(.subheadline).foregroundColor(.secondary)
                        }
                        .frame(maxWidth: .infinity).padding(40)

                    } else if let report = report {
                        SpendingReportView(report: report)

                    } else {
                        VStack(spacing: 16) {
                            Image(systemName: "chart.bar.doc.horizontal")
                                .font(.system(size: 50))
                                .foregroundColor(.blue)
                            Text("Get your AI spending report")
                                .font(.headline)
                            Text("Foundation Models analyses your transactions entirely on-device.")
                                .font(.subheadline).foregroundColor(.secondary)
                                .multilineTextAlignment(.center)
                            Button("Analyse My Spending") {
                                Task { await analyse() }
                            }
                            .buttonStyle(.borderedProminent)
                        }
                        .padding(40)
                    }
                }
                .padding()
            }
            .navigationTitle("Spending")
            .task { transactions = (try? await TransactionsAPI.shared.fetch(days: 30, category: nil)) ?? [] }
        }
    }

    private func analyse() async {
        await MainActor.run { isAnalysing = true }
        defer { Task { await MainActor.run { isAnalysing = false } } }

        do {
            for try await partial in await coordinator.analyzeSpending(transactions: transactions) {
                await MainActor.run { report = partial }
            }
        } catch is CancellationError {
            // optional: ignore
        } catch {
            // optional: surface error state
            // await MainActor.run { self.errorMessage = error.localizedDescription }
        }
    }

}

// MARK: - Spending Report View

struct SpendingReportView: View {
    let report: SpendingReport.PartiallyGenerated

    var body: some View {
        VStack(spacing: 16) {
            // Totals header
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Total Spent").font(.caption).foregroundColor(.secondary)
                    Text("$\(String(format: "%.2f", report.totalSpent ?? 0.0))")
                        .font(.largeTitle).fontWeight(.bold)
                }
                Spacer()
                VStack(alignment: .trailing, spacing: 4) {
                    Text("vs Last Period").font(.caption).foregroundColor(.secondary)
                    Text("\(report.changePercent ?? 0.0 >= 0 ? "+" : "")\(String(format: "%.1f", report.changePercent ?? 0.0))%")
                        .font(.title2).fontWeight(.semibold)
                        .foregroundColor(report.changePercent ?? 0.0 > 0 ? .red : .green)
                }
            }
            .padding()
            .background(.regularMaterial)
            .cornerRadius(16)

            // Recommendation
            HStack(spacing: 10) {
                Image(systemName: "lightbulb.fill").foregroundColor(.yellow)
                Text(report.recommendation ?? "")
                    .font(.subheadline)
            }
            .padding()
            .background(.blue.opacity(0.08))
            .cornerRadius(12)

            // Budget badge
            HStack {
                Text("Status:")
                    .font(.subheadline).foregroundColor(.secondary)
                Text(budgetLabel)
                    .font(.subheadline).fontWeight(.semibold)
                    .foregroundColor(budgetColor)
                    .padding(.horizontal, 10).padding(.vertical, 4)
                    .background(budgetColor.opacity(0.12))
                    .cornerRadius(8)
                Spacer()
            }

            // Top merchants
            VStack(alignment: .leading, spacing: 10) {
                Text("Top Merchants").font(.headline)
                let merchants = report.topMerchants ?? []
                ForEach(merchants.indices, id: \.self) { i in
                    let m = merchants[i]
                    HStack {
                        Text("\(i + 1). \(m.merchantName)")
                        Spacer()
                        Text("$\(String(format: "%.2f", m.amount ?? 0.0))")
                            .fontWeight(.semibold)
                        Text("·\(m.transactionCount)x")
                            .font(.caption).foregroundColor(.secondary)
                    }
                    .padding()
                    .background(.regularMaterial)
                    .cornerRadius(10)
                }
            }
        }
    }

    var budgetLabel: String {
        switch report.budgetStatus {
        case .some(.onTrack):    return "On Track ✓"
        case .some(.overBudget): return "Over Budget ↑"
        case .some(.underBudget): return "Under Budget ↓"
        case .none:              return "—"
        }
    }
    var budgetColor: Color {
        switch report.budgetStatus {
        case .some(.onTrack):    return .green
        case .some(.overBudget): return .red
        case .some(.underBudget): return .blue
        case .none:              return .secondary
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - Investments View
// ═══════════════════════════════════════════════════════════════

struct InvestmentsView: View {
    @Environment(FinancialAgentCoordinator.self) var coordinator
    @State private var portfolioText: String?
    @State private var isLoading = false

    var body: some View {
        NavigationStack {
            VStack(spacing: 20) {
                if isLoading {
                    ProgressView("Loading portfolio via cloud agent…")
                } else if let text = portfolioText {
                    ScrollView {
                        Text(text)
                            .font(.system(.body, design: .monospaced))
                            .padding()
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                } else {
                    VStack(spacing: 16) {
                        Image(systemName: "chart.line.uptrend.xyaxis")
                            .font(.system(size: 50)).foregroundColor(.indigo)
                        Text("J.P. Morgan Portfolio")
                            .font(.headline)
                        Text("Portfolio data is fetched via the secure cloud agent.")
                            .font(.subheadline).foregroundColor(.secondary)
                            .multilineTextAlignment(.center)
                        Button("Load Portfolio") { Task { await load() } }
                            .buttonStyle(.borderedProminent)
                            .tint(.indigo)
                    }
                    .padding(40)
                }
            }
            .navigationTitle("Investments")
        }
    }

    private func load() async {
        isLoading = true
        let response = try? await coordinator.process(
            query: "Show my investment portfolio",
            context: .current()
        )
        if case .text(let t) = response { portfolioText = t }
        isLoading = false
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - Profile View
// ═══════════════════════════════════════════════════════════════

struct ProfileView: View {
    @Environment(FinancialAgentCoordinator.self) var coordinator
    @Environment(AppSecurityService.self) var security

    var body: some View {
        NavigationStack {
            List {
                Section("AI Backend") {
                    LabeledContent("Engine", value: backendName)
                    LabeledContent("Status", value: "Active")
                }

                Section("Core ML Models") {
                    LabeledContent("Intent Classifier", value: "~8 MB")
                    LabeledContent("Fraud Detector",    value: "~3 MB")
                    LabeledContent("MiniLM Embeddings", value: "~20 MB")
                }

                Section("Security") {
                    Label("Face ID / Touch ID enabled", systemImage: "faceid")
                    Label("Secure Enclave key binding", systemImage: "lock.shield.fill")
                    Label("PCI-DSS compliant context",  systemImage: "checkmark.shield.fill")
                }

                Section {
                    Button("Sign Out", role: .destructive) {
                        security.signOut()
                    }
                }
            }
            .navigationTitle("Profile")
        }
    }

    var backendName: String {
        switch coordinator.activeBackend {
        case .foundationModels: return "Apple Foundation Models"
        case .cloud:            return "Cloud Agent"
        case .unavailable:      return "Unavailable"
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - Transfer Approval Sheet
// ═══════════════════════════════════════════════════════════════

struct TransferApprovalSheet: View {
    @Environment(FinancialAgentCoordinator.self) var coordinator
    let draft: TransferDraft
    @State private var isExecuting = false
    @State private var didSucceed  = false
    @State private var errorMsg: String?
    @Environment(\.dismiss) var dismiss

    var body: some View {
        NavigationStack {
            VStack(spacing: 28) {
                Spacer()

                // Icon
                Image(systemName: "arrow.left.arrow.right.circle.fill")
                    .font(.system(size: 60))
                    .foregroundStyle(LinearGradient(
                        colors: [.blue, .cyan], startPoint: .top, endPoint: .bottom))

                Text("Confirm Transfer")
                    .font(.title2).fontWeight(.bold)

                // Transfer details
                VStack(spacing: 0) {
                    TransferDetailRow(label: "From",   value: draft.fromAccountDisplay)
                    Divider().padding(.leading, 16)
                    TransferDetailRow(label: "To",     value: draft.toAccountDisplay)
                    Divider().padding(.leading, 16)
                    TransferDetailRow(label: "Amount", value: draft.formattedAmount, highlight: true)
                    if !draft.memo.isEmpty {
                        Divider().padding(.leading, 16)
                        TransferDetailRow(label: "Memo", value: draft.memo)
                    }
                }
                .background(.regularMaterial)
                .cornerRadius(16)

                Text("This transfer will be executed immediately after you tap Confirm.")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)

                if let err = errorMsg {
                    Text(err).font(.caption).foregroundColor(.red).multilineTextAlignment(.center)
                }

                Spacer()

                // Buttons
                VStack(spacing: 12) {
                    Button {
                        Task {
                            isExecuting = true
                            errorMsg    = nil
                            do {
                                let token = UUID().uuidString
                                try await coordinator.executeApprovedTransfer(draft, confirmationToken: token)
                                didSucceed = true
                                dismiss()
                            } catch {
                                errorMsg = error.localizedDescription
                            }
                            isExecuting = false
                        }
                    } label: {
                        Group {
                            if isExecuting {
                                ProgressView().tint(.white)
                            } else {
                                Text("Confirm Transfer — \(draft.formattedAmount)")
                                    .fontWeight(.semibold)
                            }
                        }
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(.blue)
                        .foregroundColor(.white)
                        .cornerRadius(16)
                    }
                    .disabled(isExecuting)

                    Button("Cancel", role: .cancel) {
                        coordinator.cancelTransfer()
                        dismiss()
                    }
                    .foregroundColor(.red)
                }
                .padding(.bottom, 8)
            }
            .padding(.horizontal, 24)
            .navigationBarTitleDisplayMode(.inline)
        }
    }
}

struct TransferDetailRow: View {
    let label:      String
    let value:      String
    var highlight:  Bool = false

    var body: some View {
        HStack {
            Text(label)
                .foregroundColor(.secondary)
                .frame(width: 70, alignment: .leading)
            Spacer()
            Text(value)
                .fontWeight(highlight ? .bold : .medium)
                .foregroundColor(highlight ? .primary : .primary)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 14)
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - Apple Intelligence Required View
// ═══════════════════════════════════════════════════════════════

struct AppleIntelligenceRequiredView: View {
    var body: some View {
        VStack(spacing: 28) {
            Spacer()

            Image(systemName: "apple.intelligence")
                .font(.system(size: 70))
                .foregroundStyle(LinearGradient(
                    colors: [.purple, .pink, .blue],
                    startPoint: .topLeading, endPoint: .bottomTrailing))

            VStack(spacing: 10) {
                Text("Apple Intelligence Required")
                    .font(.title2).fontWeight(.bold)
                    .multilineTextAlignment(.center)
                Text("Chase AI runs entirely on your device using Apple Foundation Models. Enable Apple Intelligence to get started.")
                    .font(.subheadline).foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)
            }

            VStack(spacing: 14) {
                Button {
                    // Deep-link to Apple Intelligence settings page
                    if let url = URL(string: "App-prefs:SIRI") {
                        UIApplication.shared.open(url)
                    }
                } label: {
                    Label("Open Settings", systemImage: "gear")
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(.blue)
                        .foregroundColor(.white)
                        .cornerRadius(16)
                }

                Text("Settings → Apple Intelligence & Siri → Turn on Apple Intelligence")
                    .font(.caption2).foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
            }
            .padding(.horizontal, 32)

            Spacer()
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - Launch View
// ═══════════════════════════════════════════════════════════════

struct LaunchView: View {
    @State private var pulse = false

    var body: some View {
        VStack(spacing: 24) {
            Image(systemName: "sparkles")
                .font(.system(size: 64))
                .foregroundStyle(LinearGradient(
                    colors: [.blue, .cyan], startPoint: .top, endPoint: .bottom))
                .scaleEffect(pulse ? 1.08 : 1.0)
                .animation(.easeInOut(duration: 1.2).repeatForever(autoreverses: true), value: pulse)

            VStack(spacing: 8) {
                Text("Chase AI").font(.largeTitle).fontWeight(.bold)
                Text("Initialising on-device AI…")
                    .font(.subheadline).foregroundColor(.secondary)
            }
            ProgressView()
        }
        .onAppear { pulse = true }
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - Biometric Auth View
// ═══════════════════════════════════════════════════════════════

struct BiometricAuthView: View {
    let authService:     AppSecurityService
    let onAuthenticated: () -> Void
    @State private var isAuthenticating = false
    @State private var error: String?

    var body: some View {
        VStack(spacing: 32) {
            Spacer()

            Image(systemName: "faceid")
                .font(.system(size: 70)).foregroundColor(.blue)

            VStack(spacing: 8) {
                Text("Secure Access").font(.title2).fontWeight(.bold)
                Text("Authenticate to access Chase AI")
                    .font(.subheadline).foregroundColor(.secondary)
            }

            if let error { Text(error).font(.caption).foregroundColor(.red) }

            Button {
                Task { await authenticate() }
            } label: {
                Label(isAuthenticating ? "Authenticating…" : "Authenticate with Face ID",
                      systemImage: "faceid")
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(.blue)
                    .foregroundColor(.white)
                    .cornerRadius(16)
            }
            .disabled(isAuthenticating)
            .padding(.horizontal, 32)

            Spacer()
        }
        .task { await authenticate() }  // auto-trigger on appear
    }

    private func authenticate() async {
        isAuthenticating = true; error = nil
        do {
            _ = try await authService.authenticateWithBiometrics()
            onAuthenticated()
        } catch let e {
            error = e.localizedDescription
        }
        isAuthenticating = false
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - Helpers
// ═══════════════════════════════════════════════════════════════

extension View {
    func cornerRadius(_ radius: CGFloat, corners: UIRectCorner) -> some View {
        clipShape(RoundedCorner(radius: radius, corners: corners))
    }
}

struct RoundedCorner: Shape {
    var radius:  CGFloat     = .infinity
    var corners: UIRectCorner = .allCorners

    func path(in rect: CGRect) -> Path {
        Path(UIBezierPath(
            roundedRect: rect,
            byRoundingCorners: corners,
            cornerRadii: CGSize(width: radius, height: radius)
        ).cgPath)
    }
}

extension FraudSignal {
    var formattedAmount: String { String(format: "$%.2f", amount) }
}

