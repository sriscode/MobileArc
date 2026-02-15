// IntentClassifier.swift
// Core ML DistilBERT intent classifier — runs on every query before Foundation Models
// ~8 MB model, <5 ms inference, 8 intent classes
// Train model: MLTraining/intent_classifier/train_intent_classifier.py

import CoreML
import Foundation

// MARK: - Intent Types

enum IntentType: String, CaseIterable {
    case balanceQuery     = "balance_query"
    case spendingAnalysis = "spending_analysis"
    case transferRequest  = "transfer_request"    // → cloud required
    case fraudReport      = "fraud_report"
    case investmentQuery  = "investment_query"    // → cloud required
    case savingsAdvice    = "savings_advice"
    case billPayment      = "bill_payment"        // → cloud required
    case general          = "general"
}

struct UserIntent {
    let type:       IntentType
    let confidence: Float

    /// These intents always route to cloud — they involve real money movement
    /// or require real-time data the on-device model cannot provide
    var requiresCloudExecution: Bool {
        [.transferRequest, .investmentQuery, .billPayment].contains(type)
    }
}

// MARK: - Classifier

class IntentClassifier {
    private var model: ChaseIntentClassifier?   // Xcode-generated typed class from .mlpackage
    private let tokenizer = LightweightBertTokenizer()
    private let maxLength = 64

    init() { loadModel() }

    private func loadModel() {
        let config = MLModelConfiguration()
        config.computeUnits = .all

        // ChaseIntentClassifier() is the Xcode-generated initialiser.
        // It's available once ChaseIntentClassifier.mlpackage is added to the Xcode target.
        // If the model hasn't been trained yet, this will fail and we fall back to keywords.
        do {
            model = try ChaseIntentClassifier(configuration: config)
            print("✅ Intent classifier loaded")
        } catch {
            print("⚠️ ChaseIntentClassifier failed to load: \(error.localizedDescription)")
            print("   Run MLTraining/intent_classifier/train_intent_classifier.py to generate it")
            model = nil
        }
    }

    func classify(_ text: String) throws -> UserIntent {
        if let model = model {
            return try classifyWithCoreML(text, model: model)
        }
        return classifyWithKeywords(text)
    }

    // MARK: - Core ML path (uses Xcode-generated typed API — no string feature name lookup)

    private func classifyWithCoreML(_ text: String, model: ChaseIntentClassifier) throws -> UserIntent {
        let tokens = tokenizer.encode(text, maxLength: maxLength)

        let inputIds  = try MLMultiArray(shape: [1, maxLength as NSNumber], dataType: .int32)
        let attnMask  = try MLMultiArray(shape: [1, maxLength as NSNumber], dataType: .int32)
        for i in 0..<tokens.inputIds.count {
            inputIds[i] = NSNumber(value: tokens.inputIds[i])
            attnMask[i] = NSNumber(value: tokens.attentionMask[i])
        }

        // Xcode-generated typed input struct — no string keys, no feature name guessing
        let input  = ChaseIntentClassifierInput(input_ids: inputIds, attention_mask: attnMask)
        let output = try model.prediction(input: input)

        // output.logits is the auto-generated typed property matching the "logits" output name
        let logits = output.logits
        let probs  = softmax(logits: logits, count: IntentType.allCases.count)
        let maxIdx = probs.indices.max(by: { probs[$0] < probs[$1] }) ?? 0
        let intent = IntentType.allCases[safe: maxIdx] ?? .general

        print("🎯 Intent: \(intent.rawValue) (confidence: \(String(format: "%.0f", probs[maxIdx] * 100))%)")
        return UserIntent(type: intent, confidence: probs[maxIdx])
    }

    private func softmax(logits: MLMultiArray, count: Int) -> [Float] {
        var v = (0..<count).map { Float(truncating: logits[$0]) }
        let m = v.max() ?? 0
        v = v.map { exp($0 - m) }
        let s = v.reduce(0, +)
        return v.map { $0 / s }
    }

    // MARK: - Keyword fallback (used during development before model is trained)

    private func classifyWithKeywords(_ text: String) -> UserIntent {
        let t = text.lowercased()
        if t.contains("balance") || t.contains("how much") || t.contains("what's in") {
            return .init(type: .balanceQuery, confidence: 0.85)
        }
        if t.contains("transfer") || t.contains("send money") || t.contains("move") || t.contains("zelle") {
            return .init(type: .transferRequest, confidence: 0.85)
        }
        if t.contains("spend") || t.contains("transaction") || t.contains("purchase") || t.contains("bought") {
            return .init(type: .spendingAnalysis, confidence: 0.80)
        }
        if t.contains("fraud") || t.contains("dispute") || t.contains("didn't make") || t.contains("unauthorized") {
            return .init(type: .fraudReport, confidence: 0.90)
        }
        if t.contains("invest") || t.contains("portfolio") || t.contains("stock") || t.contains("j.p. morgan") {
            return .init(type: .investmentQuery, confidence: 0.85)
        }
        if t.contains("save") || t.contains("saving") || t.contains("interest rate") || t.contains("apy") {
            return .init(type: .savingsAdvice, confidence: 0.80)
        }
        if t.contains("pay") || t.contains("bill") || t.contains("minimum") || t.contains("due") {
            return .init(type: .billPayment, confidence: 0.80)
        }
        return .init(type: .general, confidence: 0.50)
    }
}

// MARK: - Lightweight BERT Tokenizer
// Full production version: use swift-transformers for proper SentencePiece / WordPiece

struct BertTokens {
    let inputIds:      [Int32]
    let attentionMask: [Int32]
}

class LightweightBertTokenizer {
    private var vocab: [String: Int32] = [:]
    let clsId: Int32 = 101
    let sepId: Int32 = 102
    let padId: Int32 = 0
    let unkId: Int32 = 100

    init() { loadVocab() }

    private func loadVocab() {
        guard let path = Bundle.main.path(forResource: "bert_vocab", ofType: "txt"),
              let text = try? String(contentsOfFile: path) else {
            return }
        text.components(separatedBy: .newlines).enumerated().forEach { i, word in
            vocab[word] = Int32(i)
        }
    }

    func encode(_ text: String, maxLength: Int) -> BertTokens {
        var ids: [Int32] = [clsId]
        for word in text.lowercased().components(separatedBy: .whitespaces) {
            ids.append(vocab[word] ?? unkId)
        }
        ids.append(sepId)

        let len     = min(ids.count, maxLength)
        var inputIds = Array(ids.prefix(len))
        var mask     = Array(repeating: Int32(1), count: len)
        while inputIds.count < maxLength { inputIds.append(padId); mask.append(0) }
        return BertTokens(inputIds: inputIds, attentionMask: mask)
    }
}

// MARK: - Safe Collection Extension

extension Array {
    subscript(safe index: Index) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
