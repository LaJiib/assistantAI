//
//  ChatToolbarView.swift
//  AssistantIA
//
//  Created by JB SENET on 01/02/2026.
//

import SwiftUI

/// Toolbar view for the chat interface that displays error messages, download progress,
/// generation statistics, and model selection controls.
struct ChatToolbarView: View {
    /// View model containing the chat state and controls
    @Bindable var vm: ConversationViewModel

    var body: some View {
        // Display error message if present
        if let errorMessage = vm.errorMessage {
            ErrorView(errorMessage: errorMessage)
        }

        // Show download progress for model loading
        if let progress = vm.modelDownloadProgress, !progress.isFinished {
            DownloadProgressView(progress: progress)
        }

        // Button to clear chat history, displays generation statistics
        Button {
            vm.clear([.chat, .meta])
        } label: {
            GenerationInfoView(
                tokensPerSecond: vm.tokensPerSecond
            )
        }
    }
}
