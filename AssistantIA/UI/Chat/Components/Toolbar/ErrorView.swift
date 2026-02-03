//
//  ErrorView.swift
//  AssistantIA
//
//  Created by JB SENET on 01/02/2026.
//

import SwiftUI

struct ErrorView: View {
    let errorMessage: String

    @State private var isShowingError = false

    var body: some View {
        Button {
            isShowingError = true
        } label: {
            Image(systemName: "exclamationmark.triangle")
                .foregroundStyle(.red)
        }
        .popover(isPresented: $isShowingError, arrowEdge: .bottom) {
            Text(errorMessage)
                .padding()
        }
    }
}

#Preview {
    ErrorView(errorMessage: "Something went wrong!")
}
