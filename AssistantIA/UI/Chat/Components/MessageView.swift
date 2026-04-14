//
//  MessageView.swift
//  AssistantIA
//
//  Created by JB SENET on 01/02/2026.
//

import MarkdownUI
import SwiftUI


// MARK: - Custom Markdown Theme

extension Theme {
    static let assistant = Theme()
        // ── Titres ────────────────────────────────────────────────────────────
        .heading1 { configuration in
            configuration.label
                .markdownMargin(top: 20, bottom: 6)
                .markdownTextStyle {
                    FontWeight(.bold)
                    FontSize(.em(1.5))
                }
        }
        .heading2 { configuration in
            configuration.label
                .markdownMargin(top: 16, bottom: 4)
                .markdownTextStyle {
                    FontWeight(.semibold)
                    FontSize(.em(1.2))
                }
        }
        .heading3 { configuration in
            configuration.label
                .markdownMargin(top: 12, bottom: 2)
                .markdownTextStyle {
                    FontWeight(.semibold)
                    FontSize(.em(1.05))
                }
        }
        // ── Emphases ──────────────────────────────────────────────────────────
        .strong {
            FontWeight(.bold)
        }
        .emphasis {
            FontStyle(.italic)
            ForegroundColor(.primary.opacity(0.75))
        }
        // ── Code inline ───────────────────────────────────────────────────────
        .code {
            FontFamilyVariant(.monospaced)
            FontSize(.em(0.85))
            ForegroundColor(.secondary)
        }
        // ── Blocs de code ─────────────────────────────────────────────────────
        .codeBlock { configuration in
            ScrollView(.horizontal) {
                configuration.label
                    .relativeLineSpacing(.em(0.25))
                    .markdownTextStyle {
                        FontFamilyVariant(.monospaced)
                        FontSize(.em(0.85))
                        ForegroundColor(.secondary)
                    }
                    .padding(12)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .background(.background.secondary)
            .clipShape(.rect(cornerRadius: 8))
        }
}


// MARK: - MessageView

/// Affiche un message unique dans l'interface de chat.
/// Itère sur `message.parts` pour rendre texte, raisonnement et appels d'outils.
struct MessageView: View {
    let message: Message

    init(_ message: Message) {
        self.message = message
    }

    var body: some View {
        switch message.role {
        case .user:
            HStack {
                Spacer()
                VStack(alignment: .trailing, spacing: 8) {
                    Text(LocalizedStringKey(message.textContent))
                        .padding(.vertical, 8)
                        .padding(.horizontal, 12)
                        .background(.tint, in: .rect(cornerRadius: 16))
                        .textSelection(.enabled)
                }
            }

        case .assistant:
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 8) {
                    ForEach(Array(message.parts.enumerated()), id: \.offset) { _, part in
                        AssistantPartView(part: part)
                    }
                }
                Spacer()
            }

        case .system, .tool:
            Label(message.textContent, systemImage: "desktopcomputer")
                .font(.headline)
                .foregroundColor(.secondary)
                .frame(maxWidth: .infinity, alignment: .center)
        }
    }
}


// MARK: - AssistantPartView

/// Rendu d'une part individuelle d'un message assistant.
private struct AssistantPartView: View {
    let part: MessagePart

    var body: some View {
        switch part.type {

        // ── Texte standard : rendu Markdown complet ──────────────────────────
        case .text:
            Markdown(part.content ?? "")
                .markdownTheme(.assistant)
                .textSelection(.enabled)

        // ── Raisonnement : DisclosureGroup discret, replié par défaut ───────────
        case .reasoning:
            DisclosureGroup {
                Text(part.content ?? "")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.top, 4)
            } label: {
                Label("Réflexion", systemImage: "brain")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }
            .padding(8)
            .background(.background.secondary, in: .rect(cornerRadius: 8))

        // ── Appel d'outil : indicateur de chargement ou résultat ───────────────
        case .toolCall:
            HStack(spacing: 6) {
                if part.content == nil {
                    // Outil en cours d'exécution
                    ProgressView()
                        .scaleEffect(0.75)
                        .frame(width: 14, height: 14)
                } else {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                        .imageScale(.small)
                }
                Text(part.toolName ?? "Outil")
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
            }
            .padding(.vertical, 4)
            .padding(.horizontal, 8)
            .background(.background.secondary, in: .rect(cornerRadius: 6))

        // ── Résultat d'outil : masqué (fusionné dans la part toolCall) ──────────
        case .toolResult:
            EmptyView()
        }
    }
}


// MARK: - Previews

#Preview {
    VStack(spacing: 20) {
        MessageView(.system("Test"))

        MessageView(.user("Montre-moi une fonction Python simple"))

        MessageView(.assistant("""
            Bien sûr ! Voici un exemple :

            ```python
            def greet(name: str) -> str:
                return f"Bonjour, {name} !"

            print(greet("Alice"))
            ```
            """))

        // Message avec raisonnement + outil
        MessageView(Message(
            id: UUID(),
            role: .assistant,
            parts: [
                .reasoning("Je dois d'abord chercher les informations pertinentes..."),
                .toolCall(id: "tc1", name: "web_search"),
                .text("## Résultat\n\nD'après mes recherches, voici la **réponse**.")
            ],
            timestamp: Date()
        ))
    }
    .padding()
}
