//
//  InputBar.swift
//  AssistantIA
//
//  Created by JB SENET on 31/01/2026.
//

import SwiftUI

struct InputBar: View {
    @Binding var text: String
    let isGenerating: Bool
    let isEnabled: Bool
    let onSend: () -> Void
    
    var body: some View {
        HStack(spacing: 12) {
            TextField("Message...", text: $text, axis: .vertical)
                .textFieldStyle(.roundedBorder)
                .lineLimit(1...5)
                .disabled(isGenerating || !isEnabled)
                .onSubmit {
                    if canSend {
                        onSend()
                    }
                }
            
            Button(action: onSend) {
                Image(systemName: "arrow.up.circle.fill")
                    .font(.title2)
                    .foregroundColor(canSend ? .blue : .gray)
            }
            .disabled(!canSend)
        }
        .padding()
        .background(.background)
        .shadow(color: .black.opacity(0.05), radius: 5, y: -2)
    }
    
    private var canSend: Bool {
        !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        && !isGenerating
        && isEnabled
    }
}

#Preview {
    VStack {
        Spacer()
        InputBar(
            text: .constant(""),
            isGenerating: false,
            isEnabled: true,
            onSend: {}
        )
    }
}
