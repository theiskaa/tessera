import Foundation
import PDFKit

struct Page: Encodable {
    let page: Int
    let text: String
}

guard CommandLine.arguments.count == 2,
      let document = PDFDocument(url: URL(fileURLWithPath: CommandLine.arguments[1])) else {
    fputs("expected a readable NRCS directory PDF\n", stderr)
    exit(1)
}

let encoder = JSONEncoder()
encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
for index in 0..<document.pageCount {
    guard let text = document.page(at: index)?.string else {
        fputs("missing directory page text\n", stderr)
        exit(1)
    }
    let row = Page(page: index + 1, text: text)
    let data = try encoder.encode(row)
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data([0x0a]))
}
