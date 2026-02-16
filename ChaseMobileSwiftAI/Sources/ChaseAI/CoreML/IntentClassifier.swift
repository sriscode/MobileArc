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
    private var lrmodel: IntentClassifierLR?
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
            //            Logistic Regression model
            lrmodel = try IntentClassifierLR()
        } catch {
            print("⚠️ IntentClassifierLR failed to load: \(error.localizedDescription)")
            lrmodel = nil
        }
        
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
            // return here if you want to use coreml
            _ = try classifyWithCoreML(text, model: model)
        }
        
        if let lrmodel = lrmodel {
            let (output, conf, top) = lrmodel.predict(text)
            guard let intent = IntentType(rawValue: output) else {
                print("🎯 lrmodel Intent: fail")
                return UserIntent(type: .general, confidence: 0.8)
            }
            print("🎯 lrmodel Intent: \(intent.rawValue) (confidence: \(String(format: "%.0f", conf * 100))%)")
            return UserIntent(type: intent, confidence: conf)
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


final class IntentClassifierLR {
    struct Model: Decodable {
        let classes: [String]
        let vocab: [String: Int]
        let idf: [Float]
        let W: [[Float]]   // [C][F]
        let b: [Float]     // [C]
    }

    private let model: Model

    init(jsonFileName: String = "intent_tfidf_lr") throws {
        guard let url = Bundle.main.url(forResource: jsonFileName, withExtension: "json") else {
            throw NSError(domain: "IntentClassifierLR", code: 1, userInfo: [NSLocalizedDescriptionKey: "Model JSON not found"])
        }
        let data = try Data(contentsOf: url)
        self.model = try JSONDecoder().decode(Model.self, from: data)
    }

    // Simple tokenizer: lowercase + split on non-alphanumerics.
    // (Matches the Python token_pattern fairly closely.)
    private func tokens(_ text: String) -> [String] {
        let lower = text.lowercased()
        return lower
            .split { !$0.isLetter && !$0.isNumber && $0 != "_" }
            .map(String.init)
            .filter { !$0.isEmpty }
    }

    // Build 1-2 gram tokens
    private func ngrams(_ toks: [String]) -> [String] {
        if toks.isEmpty { return [] }
        var out = [String]()
        out.reserveCapacity(toks.count * 2)
        // unigrams
        out.append(contentsOf: toks)
        // bigrams
        if toks.count >= 2 {
            for i in 0..<(toks.count - 1) {
                out.append("\(toks[i]) \(toks[i+1])")
            }
        }
        return out
    }

    /// Returns (intent, confidence, topK)
    func predict(_ text: String, topK: Int = 3) -> (String, Float, [(String, Float)]) {
        let toks = ngrams(tokens(text))

        // term counts for known vocab
        var counts: [Int: Int] = [:]
        counts.reserveCapacity(toks.count)

        for t in toks {
            if let idx = model.vocab[t] {
                counts[idx, default: 0] += 1
            }
        }

        if counts.isEmpty {
            return ("general", 0.0, [("general", 0.0)])
        }

        // TF-IDF vector in sparse form: x[j] = tf(j) * idf[j]
        // Use tf = raw count (good enough for this small model)
        // Optionally normalize L2 like sklearn does: we’ll do L2 normalization.
        var x: [Int: Float] = [:]
        x.reserveCapacity(counts.count)

        var norm2: Float = 0
        for (j, c) in counts {
            let v = Float(c) * model.idf[j]
            x[j] = v
            norm2 += v * v
        }
        let norm = sqrt(norm2)
        if norm > 0 {
            for (j, v) in x {
                x[j] = v / norm
            }
        }

        // Compute logits = W*x + b
        let C = model.classes.count
        var logits = Array(repeating: Float(0), count: C)

        for c in 0..<C {
            var s = model.b[c]
            let wRow = model.W[c]
            for (j, v) in x {
                s += wRow[j] * v
            }
            logits[c] = s
        }

        // Softmax
        let maxLogit = logits.max() ?? 0
        var exps = logits.map { expf($0 - maxLogit) }
        let sumExp = exps.reduce(0, +)
        let probs = exps.map { $0 / sumExp }

        // TopK
        let ranked = probs.enumerated()
            .sorted { $0.element > $1.element }
            .prefix(max(1, topK))
            .map { (model.classes[$0.offset], $0.element) }

        let best = ranked[0]
        return (best.0, best.1, ranked)
    }
}


// MARK: - Safe Collection Extension

extension Array {
    subscript(safe index: Index) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
