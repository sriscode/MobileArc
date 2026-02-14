// FraudDetector.swift
// Core ML IsolationForest fraud detector — runs on every transaction event
// ~3 MB model, <1 ms inference, 8 engineered features
// Train model: MLTraining/fraud_detector/train_fraud_model.py

import CoreML
import CoreLocation
import Foundation

// MARK: - Transaction History (used to compute relative features)

@MainActor
class TransactionHistory {
    static let shared = TransactionHistory()

    private var recent: [Transaction] = []
    private var lastLocation: CLLocation?
    
    private init() {}

    func update(transactions: [Transaction], location: CLLocation? = nil) {
        recent        = Array(transactions.prefix(100))
        lastLocation  = location
    }

    // Count transactions within the last N hours
    func velocity(hours: Int) -> Int {
        let cutoff = Date().addingTimeInterval(-Double(hours * 3_600))
        return recent.filter { $0.timestamp > cutoff }.count
    }

    // Distance in km from the last known transaction location
    func distanceKm(from coordinate: CLLocationCoordinate2D?) -> Double {
        guard let coord = coordinate, let last = lastLocation else { return 0 }
        return CLLocation(latitude: coord.latitude, longitude: coord.longitude)
            .distance(from: last) / 1_000
    }

    // Sigmoid-normalized amount relative to account history (0–1 scale)
    func normalizedAmount(_ amount: Double) -> Double {
        guard !recent.isEmpty else { return 0.5 }
        let amounts = recent.map(\.amount)
        let mean    = amounts.reduce(0, +) / Double(amounts.count)
        let variance = amounts.map { pow($0 - mean, 2) }.reduce(0, +) / Double(amounts.count)
        let stdDev  = sqrt(variance)
        guard stdDev > 0 else { return 0.5 }
        let z = (amount - mean) / stdDev
        return 1.0 / (1.0 + exp(-z))   // sigmoid
    }
}

// MARK: - Fraud Detector

class FraudDetector: @unchecked Sendable {
    private var model: MLModel?

    // Thresholds (IsolationForest: lower score = more anomalous)
    private let suspiciousThreshold: Float = -0.15
    private let criticalThreshold:   Float = -0.30

    init() { loadModel() }

    private func loadModel() {
        guard let url = Bundle.main.url(
            forResource: "ChaseAnomalyDetector", withExtension: "mlmodelc"
        ) else {
            print("⚠️ ChaseAnomalyDetector.mlmodelc not found — using rule-based fallback")
            print("   Run MLTraining/fraud_detector/train_fraud_model.py to generate it")
            return
        }
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        model = try? MLModel(contentsOf: url, configuration: config)
        print("✅ Fraud detector loaded")
    }

    // MARK: - Analyse a single transaction

    func analyze(_ txn: Transaction) async throws -> FraudSignal {
        return model != nil
            ? try await analyzeWithCoreML(txn)
            : await analyzeWithRules(txn)
    }

    private func analyzeWithCoreML(_ txn: Transaction) async throws -> FraudSignal {
        guard let model = model else { return await analyzeWithRules(txn) }

        let features = try MLMultiArray(shape: [8], dataType: .float32)
        
        // Get transaction history data from main actor
        let (normalizedAmount, distanceKm, velocity1h, velocity24h) = await MainActor.run {
            let history = TransactionHistory.shared
            return (
                history.normalizedAmount(txn.amount),
                history.distanceKm(from: txn.coordinate),
                Double(history.velocity(hours: 1)),
                Double(history.velocity(hours: 24))
            )
        }
        
        // Feature 0: Amount normalised to account history (sigmoid z-score)
        features[0] = NSNumber(value: normalizedAmount)
        // Feature 1-2: Cyclical hour encoding
        let hour    = Double(Calendar.current.component(.hour, from: txn.timestamp))
        features[1] = NSNumber(value: sin(2 * .pi * hour / 24))
        features[2] = NSNumber(value: cos(2 * .pi * hour / 24))
        // Feature 3: Merchant category code risk score
        features[3] = NSNumber(value: txn.merchantCategoryRiskScore)
        // Feature 4: Distance from last transaction (km)
        features[4] = NSNumber(value: distanceKm)
        // Feature 5: Transaction count in last 1 hour
        features[5] = NSNumber(value: velocity1h)
        // Feature 6: Transaction count in last 24 hours
        features[6] = NSNumber(value: velocity24h)
        // Feature 7: Card-not-present flag
        features[7] = NSNumber(value: txn.isCardNotPresent ? 1.0 : 0.0)

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "features": MLFeatureValue(multiArray: features)
        ])
        let output = try await model.prediction(from: input)
        let scoreValue = output.featureValue(for: "anomaly_score")
        let score: Float
        if let v = scoreValue?.doubleValue {
            score = Float(v)
        } else if let v = scoreValue?.int64Value {
            score = Float(v)
        } else if let v = scoreValue?.multiArrayValue?.firstObject as? NSNumber {
            score = v.floatValue
        } else {
            score = 0
        }

        return FraudSignal(
            transactionId:           txn.id,
            merchant:                txn.merchant,
            amount:                  txn.amount,
            anomalyScore:            score,
            isSuspicious:            score < suspiciousThreshold,
            confidence:              min(abs(score) * 2, 1.0),
            requiresImmediateReview: score < criticalThreshold
        )
    }

    private func analyzeWithRules(_ txn: Transaction) async -> FraudSignal {
        // Simple rule-based fallback for development without trained model
        var risk: Float = 0
        if txn.amount > 500  { risk -= 0.05 }
        if txn.amount > 2_000 { risk -= 0.10 }
        if txn.isCardNotPresent { risk -= 0.08 }
        
        // Get transaction history data from main actor
        let (velocity1h, distanceKm) = await MainActor.run {
            let history = TransactionHistory.shared
            return (
                history.velocity(hours: 1),
                history.distanceKm(from: txn.coordinate)
            )
        }
        
        if velocity1h > 5 { risk -= 0.12 }
        if distanceKm > 500 { risk -= 0.20 }
        if txn.merchantCategoryRiskScore > 0.7 { risk -= 0.10 }

        return FraudSignal(
            transactionId:           txn.id,
            merchant:                txn.merchant,
            amount:                  txn.amount,
            anomalyScore:            risk,
            isSuspicious:            risk < suspiciousThreshold,
            confidence:              min(abs(risk) * 2, 1.0),
            requiresImmediateReview: risk < criticalThreshold
        )
    }

    // MARK: - Batch scan (called by coordinator in parallel)
    // Returns the FraudSignal for the first suspicious transaction found, or nil.
    func scanLatest(_ transactions: [Transaction]) async -> FraudSignal? {
        for transaction in transactions {
            if let result = try? await analyze(transaction), result.isSuspicious {
                return result
            }
        }
        return nil
    }
}

