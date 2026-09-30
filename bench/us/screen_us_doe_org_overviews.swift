import CryptoKit
import Foundation
import PDFKit

let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
let capture = root.appendingPathComponent("data/raw/us-doe-organization-overviews-2024-v1")
let pdf = capture.appendingPathComponent("source.pdf")
let sourceManifest = capture.appendingPathComponent("manifest.json")
let output = root.appendingPathComponent("data/raw/candidates/us-doe-org-overviews-v1")
let packet = output.appendingPathComponent("blind-v1.jsonl")
let packetManifest = output.appendingPathComponent("blind-v1.manifest.json")

func hash(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

func json(_ value: [String: Any]) throws -> Data {
    try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys])
}

let pdfData = try Data(contentsOf: pdf)
let pdfHash = hash(pdfData)
guard let source = try JSONSerialization.jsonObject(with: Data(contentsOf: sourceManifest)) as? [String: Any] else {
    fatalError("DOE source manifest is not an object")
}
guard pdfData.starts(with: Data("%PDF-".utf8)),
      source["pdf_sha256"] as? String == pdfHash,
      source["pages"] as? Int == 170,
      source["training_eligible"] as? Bool == false,
      source["evaluation_eligible"] as? Bool == false,
      let sourceURL = source["source_url"] as? String,
      let document = PDFDocument(url: pdf), document.pageCount == 170 else {
    fatalError("DOE source PDF or capture manifest changed")
}

let groups = [("infrastructure", 6...41), ("science", 42...62),
              ("nuclear_security", 63...102), ("staff", 103...170)]
var rows = [[String: Any]]()
var counts = [String: Int]()
for (group, pages) in groups {
    for pageNumber in pages {
        guard let pageText = document.page(at: pageNumber - 1)?.string,
              let start = pageText.range(of: "Supporting the DOE Mission"),
              let end = pageText.range(of: "Mission Statement", range: start.upperBound..<pageText.endIndex) else {
            continue
        }
        let body = String(pageText[start.lowerBound..<end.lowerBound])
        let words = body.split(whereSeparator: { $0.isWhitespace }).count
        guard (100...450).contains(words), !body.contains("\u{FFFD}") else { continue }
        let startByte = String(pageText[..<start.lowerBound]).utf8.count
        let endByte = String(pageText[..<end.lowerBound]).utf8.count
        guard Data(pageText.utf8)[startByte..<endByte] == Data(body.utf8) else {
            fatalError("PDF page byte window changed on page \(pageNumber)")
        }
        rows.append([
            "name": "us-doe-org-overviews-2024-page-\(pageNumber)",
            "country": "US",
            "input": body,
            "source": "us-doe-organization-overviews-2024",
            "source_url": sourceURL,
            "source_document_key": "energy.gov:2024-organization-overviews",
            "source_group": "energy.gov:2024-organization-overviews",
            "source_page": pageNumber,
            "source_start_byte": startByte,
            "source_end_byte": endByte,
            "source_pdf_sha256": pdfHash,
            "stratum": group
        ])
        counts[group, default: 0] += 1
    }
}
guard rows.count == 24,
      counts == ["infrastructure": 8, "science": 3, "nuclear_security": 2, "staff": 11] else {
    fatalError("DOE blind packet selection changed: \(counts)")
}
let packetData = try rows.reduce(into: Data()) { result, row in
    result.append(try json(row))
    result.append(0x0A)
}
let manifest: [String: Any] = [
    "kind": "us_doe_org_overviews_blind_v1",
    "training_eligible": false,
    "evaluation_eligible": false,
    "cases": rows.count,
    "source_documents": 1,
    "strata": counts,
    "selection": "mission prose of 100-450 whitespace words without replacement characters",
    "source_manifest_sha256": hash(try Data(contentsOf: sourceManifest)),
    "source_pdf_sha256": pdfHash,
    "sha256": hash(packetData)
]
let manifestData = try json(manifest) + Data([0x0A])
try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
if FileManager.default.fileExists(atPath: packet.path), try Data(contentsOf: packet) != packetData {
    fatalError("frozen DOE blind packet changed")
}
if FileManager.default.fileExists(atPath: packetManifest.path),
   try Data(contentsOf: packetManifest) != manifestData {
    fatalError("frozen DOE blind packet manifest changed")
}
if !FileManager.default.fileExists(atPath: packet.path) { try packetData.write(to: packet) }
if !FileManager.default.fileExists(atPath: packetManifest.path) { try manifestData.write(to: packetManifest) }
print("\(rows.count) source-pinned DOE passages reserved for blind evaluation review")
