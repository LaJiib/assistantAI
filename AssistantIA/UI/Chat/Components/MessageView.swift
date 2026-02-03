//
//  MessageView.swift
//  AssistantIA
//
//  Created by JB SENET on 01/02/2026.
//

import AVKit
import SwiftUI


enum ContentSegment {
    case text(String)
    case code(String, language: String?)
}

extension String {
    /// Découpe une chaîne en segments alternant texte normal et blocs de code fencés.
    /// Les blocs de code sont délimités par ``` (optionnellement suivis d'un nom de langage).
    func parsedSegments() -> [ContentSegment] {
        var segments: [ContentSegment] = []
        var remaining = self[self.startIndex...]
        
        while let openRange = remaining.range(of: "```") {
            // Texte avant le bloc de code
            let textBefore = String(remaining[remaining.startIndex..<openRange.lowerBound])
            if !textBefore.isEmpty {
                segments.append(.text(textBefore))
            }
            
            // Avancer après le ```
            let afterOpen = remaining[openRange.upperBound...]
            
            // Chercher le ``` de fermeture
            guard let closeRange = afterOpen.range(of: "```") else {
                // Pas de fermeture trouvée — le reste est du code (bloc non fermé)
                segments.append(.code(String(afterOpen), language: nil))
                remaining = remaining[remaining.endIndex...]
                break
            }
            
            // Extraire le contenu entre les deux ```
            let codeContent = String(afterOpen[afterOpen.startIndex..<closeRange.lowerBound])
            
            // Le premier mot de la première ligne peut être le nom du langage
            let lines = codeContent.split(separator: "\n", maxSplits: 1, omittingEmptySubsequences: false)
            var language: String? = nil
            var actualCode = codeContent
            
            if let firstLine = lines.first {
                let candidate = String(firstLine).trimmingCharacters(in: .whitespaces)
                if !candidate.isEmpty && !candidate.contains(" ") {
                    // C'est un mot seul sur la première ligne = nom du langage
                    language = candidate
                    // Le code commence après cette première ligne
                    actualCode = lines.count > 1 ? String(lines[1]) : ""
                }
            }
            
            segments.append(.code(actualCode, language: language))
            
            // Avancer après le ``` de fermeture
            remaining = afterOpen[closeRange.upperBound...]
        }
        
        // Texte restant après le dernier bloc de code
        let lastText = String(remaining)
        if !lastText.isEmpty {
            segments.append(.text(lastText))
        }
        
        // Si aucun segment n'a été créé (pas de ``` du tout), retourner le texte entier
        if segments.isEmpty {
            segments.append(.text(self))
        }
        
        return segments
    }
}

extension ContentSegment: CustomStringConvertible {
    var description: String {
        switch self {
        case .text(let text):
            return "text:\(text.prefix(50))"
        case .code(let code, let language):
            return "code:\(language ?? "unknown"):\(code.prefix(50))"
        }
    }
}

/// A view that displays a single message in the chat interface.
/// Supports different message roles (user, assistant, system) and media attachments.
struct MessageView: View {
    /// The message to be displayed
    let message: Message

    /// Creates a message view
    /// - Parameter message: The message model to display
    init(_ message: Message) {
        self.message = message
    }

    var body: some View {
        switch message.role {
        case .user:
            // User messages are right-aligned with blue background
            HStack {
                Spacer()
                VStack(alignment: .trailing, spacing: 8) {
                    // Message content with tinted background.
                    // LocalizedStringKey used to trigger default handling of markdown content.
                    Text(LocalizedStringKey(message.content))
                        .padding(.vertical, 8)
                        .padding(.horizontal, 12)
                        .background(.tint, in: .rect(cornerRadius: 16))
                        .textSelection(.enabled)
                }
            }

        case .assistant:
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 8) {
                    ForEach(message.content.parsedSegments(), id: \.description) { segment in
                        switch segment {
                        case .text(let text):
                            Text(LocalizedStringKey(text))
                                .textSelection(.enabled)
                                
                        case .code(let code, _):
                            ScrollView(.horizontal) {
                                Text(code)
                                    .font(.system(size: 13, weight: .regular, design: .monospaced))
                                    .foregroundStyle(.secondary)
                                    .padding(12)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .textSelection(.enabled)
                            }
                            .background(.background.secondary)
                            .clipShape(.rect(cornerRadius: 8))
                        }
                    }
                }
                Spacer()
            }

        case .system:
            // System messages are centered with computer icon
            Label(message.content, systemImage: "desktopcomputer")
                .font(.headline)
                .foregroundColor(.secondary)
                .frame(maxWidth: .infinity, alignment: .center)
        }
    }
}



#Preview {
    VStack(spacing: 20) {
        MessageView(.system(Constants.defaultSystemPrompt))

        MessageView(.user("Montre-moi une fonction Python simple"))

        MessageView(.assistant("""
            Bien sûr ! Voici un exemple :

            ```python
            def greet(name: str) -> str:
                return f"Bonjour, {name} !"

            print(greet("Alice"))
        """))
    }
    .padding()
}
