//
//  SetupView.swift
//  AssistantIA
//
//  Created by JB SENET on 01/02/2026.
//

import SwiftUI

struct SetupView: View {
    @Bindable var modelStateManager: ModelStateManager
    
    var body: some View {
        VStack(spacing: 30) {
            Text("Welcome to AssistantIA")
                .font(.largeTitle)
                .bold()
            
            VStack(spacing: 10) {
                Text("Select your MLX model folder")
                    .font(.headline)
                
                Text(modelStateManager.stateDescription)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
            
            Button {
                _ = modelStateManager.requestModelFolder()
            } label: {
                Label("Select Model Folder", systemImage: "folder")
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            
            if case .configured = modelStateManager.modelState {
                Button {
                    Task {
                        try? await modelStateManager.loadModel()
                    }
                } label: {
                    Label("Load Model", systemImage: "arrow.down.circle")
                }
                .buttonStyle(.bordered)
            }
            
            if case .loading = modelStateManager.modelState {
                ProgressView("Loading model...")
            }
        }
        .padding(40)
        .frame(minWidth: 500, minHeight: 400)
    }
}
