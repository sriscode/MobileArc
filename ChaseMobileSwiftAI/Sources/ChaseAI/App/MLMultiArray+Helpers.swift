import CoreML

extension MLMultiArray {
    /// Returns the first element of the MLMultiArray as NSNumber, or nil if the array is empty.
    var firstObject: NSNumber? {
        guard self.count > 0 else { return nil }
        return self[0]
    }
}
