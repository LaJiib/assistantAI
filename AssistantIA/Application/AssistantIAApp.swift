//
//  AssistantIAApp.swift
//  AssistantIA
//
//  Created by JB SENET on 31/01/2026.
//

import SwiftUI

@main
struct AssistantIAApp: App {
    @State private var mlxService: MLXService
    @State private var conversationManager: ConversationManager
    @State private var modelStateManager: ModelStateManager
    init() {
        let service = MLXService()
        let manager = ConversationManager(mlxService: service)

        _mlxService = State(initialValue: service)
        _conversationManager = State(initialValue: manager)
        _modelStateManager = State(initialValue: ModelStateManager(mlxService: service, conversationManager: manager))
    }
    var body: some Scene {
        WindowGroup {
            RootView(modelStateManager: modelStateManager,
                     conversationManager: conversationManager)
        }
    }
}
