// ChaseAIApp.swift
// App entry point — iOS 26+ required
// Flow: Launch → Biometric auth → AI init → Home (or Apple Intelligence required screen)

import SwiftUI
import FoundationModels

@main
struct ChaseAIApp: App {

    @State private var coordinator  = FinancialAgentCoordinator()
    @State private var authService  = AppSecurityService()
    @State private var isAuthenticated = false
    @State private var isLoading       = true

    var body: some Scene {
        WindowGroup {
            Group {
                if isLoading {
                    LaunchView()

                } else if !isAuthenticated {
                    BiometricAuthView(authService: authService) {
                        isAuthenticated = true
                    }

                } else if coordinator.activeBackend == .unavailable {
                    // iOS 26 device but Apple Intelligence is off in Settings
                    AppleIntelligenceRequiredView()

                } else {
                    MainTabView()
                        .environment(coordinator)
                        .environment(authService)
                }
            }
            .task {
                await coordinator.initialize()
                isLoading = false
            }
        }
    }
}

