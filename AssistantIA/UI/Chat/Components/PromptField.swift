//
//  PromptField.swift
//  AssistantIA
//
//  Created by JB SENET on 01/02/2026.
//

import SwiftUI

struct PromptField: View {
    @Binding var prompt: String
    @State private var task: Task<Void, Never>?
    @Binding var options: [String: Bool]

    let sendButtonAction: () async -> Void
    let canSend: Bool

    var body: some View {
        VStack(spacing: 8) {
            HStack {
                Toggle(isOn: Binding(
                    get: { options["think"] ?? false },
                    set: { options["think"] = $0 }
                )) {
                    Label("Mode Réflexion", systemImage:"brain.head.profile")
                                .font(.caption)
                }
                .toggleStyle(.button)
                .buttonStyle(.bordered)
                .tint(options["think"] == true ? .purple : .gray)
                        
                Spacer()
            }
            .padding(.horizontal)
            
            HStack {
                TextField("Prompt", text: $prompt)
                    .textFieldStyle(.roundedBorder)
                
                Button {
                    if isRunning {
                        task?.cancel()
                        removeTask()
                    } else {
                        task = Task {
                            await sendButtonAction()
                            removeTask()
                        }
                    }
                } label: {
                    Image(systemName: isRunning ? "stop.circle.fill" :  "paperplane.fill")
                }
                .keyboardShortcut(isRunning ? .cancelAction :   .defaultAction)
                .disabled(!canSend && !isRunning)
            }
        }
    }

    private var isRunning: Bool {
        task != nil && !(task!.isCancelled)
    }

    private func removeTask() {
        task = nil
    }
}

#Preview {
    PromptField(
        prompt: .constant(""),
        options: .constant(["think": false]),
        sendButtonAction: { },
        canSend: true
    )
}
