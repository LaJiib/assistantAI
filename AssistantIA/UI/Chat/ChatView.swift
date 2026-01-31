//
//  ChatView.swift
//  AssistantIA
//
//  Created by JB SENET on 31/01/2026.
//

import SwiftUI

struct ChatView: View {
    @State private var viewModel: ChatViewModel
    
    init() {
        let mlxService = MLXService()
        _viewModel = State(initialValue: ChatViewModel(mlxService: mlxService))
    }
    
    var body: some View {
        VStack(spacing: 0) {
            // Status bar
            statusBar
            
            // Liste des messages
            messageList
            
            // Barre d'input
            InputBar(
                text: $viewModel.prompt,
                isGenerating: viewModel.isGenerating,
                isEnabled: true,
                onSend: {
                    Task {
                        await viewModel.generate()
                    }
                }
            )
        }
    }
    
    private var statusBar: some View {
        HStack {
            if viewModel.isGenerating {
                ProgressView()
                    .scaleEffect(0.8)
                Text("Génération en cours...")
                    .font(.caption)
                    .foregroundColor(.secondary)
            } else {
                Circle()
                    .fill(Color.green)
                    .frame(width: 8, height: 8)
                Text("Prêt")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            if let error = viewModel.errorMessage {
                Spacer()
                Circle()
                    .fill(Color.red)
                    .frame(width: 8, height: 8)
                Text("Erreur: \(error)")
                    .font(.caption)
                    .foregroundColor(.red)
                    .lineLimit(1)
            }
            
            Spacer()
        }
        .padding(.horizontal)
        .padding(.vertical, 8)
        .background(Color.gray.opacity(0.1))
    }
    
    private var messageList: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(spacing: 12) {
                    ForEach(viewModel.messages) { message in
                        MessageBubble(message: message)
                            .id(message.id)
                    }
                    
                    if viewModel.isGenerating && viewModel.messages.isEmpty {
                        HStack {
                            ProgressView()
                            Text("Réflexion en cours...")
                                .foregroundColor(.secondary)
                        }
                    }
                }
                .padding()
            }
            .onChange(of: viewModel.messages.count) { _ in
                if let lastMessage = viewModel.messages.last {
                    withAnimation {
                        proxy.scrollTo(lastMessage.id, anchor: .bottom)
                    }
                }
            }
        }
    }
}

#Preview {
    ChatView()
}
