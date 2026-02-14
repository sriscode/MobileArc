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
    private var model: MLModel?
    private let tokenizer = LightweightBertTokenizer()
    private let maxLength = 64

    init() { loadModel() }

    private func loadModel() {
        guard let url = Bundle.main.url(
            forResource: "ChaseIntentClassifier", withExtension: "mlmodelc"
        ) else {
            print("⚠️ ChaseIntentClassifier.mlmodelc not found — using keyword fallback")
            print("   Run MLTraining/intent_classifier/train_intent_classifier.py to generate it")
            return
        }
        let config = MLModelConfiguration()
        config.computeUnits = .all          // Use ANE when available
        model = try? MLModel(contentsOf: url, configuration: config)
        print("✅ Intent classifier loaded")
    }

    func classify(_ text: String) throws -> UserIntent {
        if let model = model {
            return try classifyWithCoreML(text, model: model)
        }
        return classifyWithKeywords(text)
    }

    // MARK: - Core ML path

    private func classifyWithCoreML(_ text: String, model: MLModel) throws -> UserIntent {
        let tokens = tokenizer.encode(text, maxLength: maxLength)

        let inputIds  = try MLMultiArray(shape: [1, maxLength as NSNumber], dataType: .int32)
        let attnMask  = try MLMultiArray(shape: [1, maxLength as NSNumber], dataType: .int32)
        for i in 0..<tokens.inputIds.count {
            inputIds[i] = NSNumber(value: tokens.inputIds[i])
            attnMask[i] = NSNumber(value: tokens.attentionMask[i])
        }

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids":      MLFeatureValue(multiArray: inputIds),
            "attention_mask": MLFeatureValue(multiArray: attnMask)
        ])
        let output = try model.prediction(from: input)

        guard let logits = output.featureValue(for: "logits")?.multiArrayValue else {
            return classifyWithKeywords(text)
        }

        let probs  = softmax(logits: logits, count: IntentType.allCases.count)
        let maxIdx = probs.indices.max(by: { probs[$0] < probs[$1] }) ?? 0
        let intent = IntentType.allCases[safe: maxIdx] ?? .general
        return UserIntent(type: intent, confidence: probs[maxIdx])
    }

    private func softmax(logits: MLMultiArray, count: Int) -> [Float] {
        var values: [Float] = []
        values.reserveCapacity(count)
        for i in 0..<count {
            let v: Float
            if logits.dataType == .float32 {
                v = logits[i].floatValue
            } else if logits.dataType == .double {
                v = Float(truncating: logits[i])
            } else {
                v = Float(truncating: logits[i])
            }
            values.append(v)
        }
        let maxVal = values.max() ?? 0
        let exps = values.map { exp($0 - maxVal) }
        let sum = exps.reduce(0, +)
        return exps.map { $0 / (sum == 0 ? 1 : sum) }
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
              let text = try? String(contentsOfFile: path) else { return }
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
