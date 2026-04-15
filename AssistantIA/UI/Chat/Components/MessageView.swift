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

        // ── Appel d'outil : badge de chargement ou aperçu du résultat ──────────
        case .toolCall:
            ToolCallView(part: part)

        // ── Résultat d'outil : masqué (fusionné dans la part toolCall) ──────────
        case .toolResult:
            EmptyView()
        }
    }
}


// MARK: - ToolCallView

/// Rendu adaptatif d'un appel d'outil :
///   • En cours     → spinner + nom
///   • Complété sans preview  → checkmark + nom
///   • searchResults preview  → DisclosureGroup avec liste titre/URL
///   • webpage preview        → DisclosureGroup avec URL + nb mots
private struct ToolCallView: View {
    let part: MessagePart
    @State private var isExpanded = false

    private var isCompleted: Bool { part.isCompleted == true }

    var body: some View {
        Group {
            if !isCompleted {
                loadingBadge
            } else if let preview = part.preview, preview.previewType == "searchResults" {
                searchResultsPreview(preview)
            } else if let preview = part.preview, preview.previewType == "webpage" {
                webpagePreview(preview)
            } else {
                completedBadge
            }
        }
    }

    // ── Spinner (outil en cours) ─────────────────────────────────────────────
    private var loadingBadge: some View {
        Label {
            Text(part.toolName ?? "Outil")
                .font(.caption2.monospaced())
                .foregroundStyle(.secondary)
        } icon: {
            ProgressView()
                .scaleEffect(0.65)
                .frame(width: 12, height: 12)
        }
        .padding(.vertical, 3)
        .padding(.horizontal, 6)
        .background(.background.secondary, in: .rect(cornerRadius: 6))
    }

    // ── Checkmark simple (complété, pas de preview) ──────────────────────────
    private var completedBadge: some View {
        Label {
            Text(part.toolName ?? "Outil")
                .font(.caption2.monospaced())
                .foregroundStyle(.secondary)
        } icon: {
            Image(systemName: "checkmark.circle.fill")
                .foregroundStyle(.green)
                .imageScale(.small)
        }
        .padding(.vertical, 3)
        .padding(.horizontal, 6)
        .background(.background.secondary, in: .rect(cornerRadius: 6))
    }

    // ── Résultats de recherche ───────────────────────────────────────────────
    @ViewBuilder
    private func searchResultsPreview(_ preview: ToolPreview) -> some View {
        let results = preview.results ?? []
        let n = preview.count ?? results.count
        DisclosureGroup(isExpanded: $isExpanded) {
            VStack(alignment: .leading, spacing: 0) {
                ForEach(Array(results.enumerated()), id: \.offset) { idx, result in
                    VStack(alignment: .leading, spacing: 0) {
                        Text(result.title)
                            .font(.caption2)
                            .foregroundStyle(.primary)
                            .lineLimit(1)
                        Text(result.url)
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                            .lineLimit(1)
                    }
                    .padding(.vertical, 3)
                    if idx < results.count - 1 { Divider() }
                }
            }
            .padding(.top, 2)
        } label: {
            HStack(spacing: 4) {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundStyle(.green)
                    .imageScale(.small)
                Text(part.toolName ?? "Outil")
                    .font(.caption2.monospaced())
                    .foregroundStyle(.secondary)
                Spacer()
                Text("\(n) résultat\(n > 1 ? "s" : "")")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
        }
        .padding(6)
        .background(.background.secondary, in: .rect(cornerRadius: 8))
    }

    // ── Aperçu page web ──────────────────────────────────────────────────────
    @ViewBuilder
    private func webpagePreview(_ preview: ToolPreview) -> some View {
        let urlString = preview.url ?? ""
        let host = URL(string: urlString)?.host ?? urlString
        DisclosureGroup(isExpanded: $isExpanded) {
            VStack(alignment: .leading, spacing: 1) {
                Text(urlString)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .lineLimit(2)
                if let wc = preview.wordCount {
                    Text("\(wc.formatted()) mots\(preview.truncated == true ? " · tronqué" : "")")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
            }
            .padding(.top, 2)
        } label: {
            HStack(spacing: 4) {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundStyle(.green)
                    .imageScale(.small)
                Text(part.toolName ?? "Outil")
                    .font(.caption2.monospaced())
                    .foregroundStyle(.secondary)
                Spacer()
                Text(host)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .lineLimit(1)
            }
        }
        .padding(6)
        .background(.background.secondary, in: .rect(cornerRadius: 8))
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

        // Tool call en cours
        MessageView(Message(
            id: UUID(), role: .assistant,
            parts: [.toolCall(id: "tc1", name: "web_search")],
            timestamp: Date()
        ))

        // Tool call complété avec résultats de recherche
        MessageView(Message(
            id: UUID(), role: .assistant,
            parts: [
                MessagePart(
                    type: .toolCall, content: nil, toolCallId: "tc2", toolName: "web_search",
                    preview: ToolPreview(
                        previewType: "searchResults", query: "pydantic ai", count: 3,
                        results: [
                            SearchResultItem(title: "Pydantic AI Docs", url: "https://ai.pydantic.dev"),
                            SearchResultItem(title: "GitHub — pydantic/pydantic-ai", url: "https://github.com/pydantic/pydantic-ai"),
                        ],
                        url: nil, wordCount: nil, truncated: nil
                    ),
                    isCompleted: true
                ),
                .text("D'après mes recherches, voici la **réponse**.")
            ],
            timestamp: Date()
        ))

        // Tool call fetch_webpage complété
        MessageView(Message(
            id: UUID(), role: .assistant,
            parts: [
                MessagePart(
                    type: .toolCall, content: nil, toolCallId: "tc3", toolName: "fetch_webpage",
                    preview: ToolPreview(
                        previewType: "webpage", query: nil, count: nil, results: nil,
                        url: "https://ai.pydantic.dev/agents/", wordCount: 2_341, truncated: false
                    ),
                    isCompleted: true
                ),
            ],
            timestamp: Date()
        ))
    }
    .padding()
}
