//
//  GenerationInfoView.swift
//  AssistantIA
//
//  Created by JB SENET on 01/02/2026.
//

import SwiftUI

struct GenerationInfoView: View {
    let tokensPerSecond: Double

    var body: some View {
        Text("\(tokensPerSecond, format: .number.precision(.fractionLength(2))) tokens/s")
    }
}

#Preview {
    GenerationInfoView(tokensPerSecond: 58.5834)
}
