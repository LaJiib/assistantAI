//
//  BookmarkManager.swift
//  AssistantIA
//
//  Created by JB SENET on 01/02/2026.
//

import Foundation
import AppKit

enum BookmarkError: LocalizedError {
    case accessDenied
    case bookmarkStale
    
    var errorDescription: String? {
        switch self {
        case .accessDenied:
            return "Access to the folder was denied. Please select it again."
        case .bookmarkStale:
            return "The bookmark is no longer valid."
        }
    }
}

class BookmarkManager {
    private let bookmarkKey: String = "modelFolderBookmark"
    
    func saveBookmark(for url: URL) throws {
        let isAccessing = url.startAccessingSecurityScopedResource()
        defer {url.stopAccessingSecurityScopedResource()
        }
        let bookmarkData = try url.bookmarkData(options: .withSecurityScope, includingResourceValuesForKeys: nil, relativeTo: nil)
        UserDefaults.standard.set(bookmarkData, forKey: bookmarkKey)
    }
    
    func loadBookmark() throws -> URL? {
        guard let bookmarkData = UserDefaults.standard.data(forKey: bookmarkKey) else {
            return nil
        }
        var isStale = false
        let url = try URL(resolvingBookmarkData: bookmarkData,
            options: .withSecurityScope, relativeTo: nil, bookmarkDataIsStale: &isStale)
        // Démarrer l'accès security-scoped
        let didStartAccessing = url.startAccessingSecurityScopedResource()
        guard didStartAccessing else {
            throw BookmarkError.accessDenied
        }
        if isStale {
            try saveBookmark(for: url)
        }
        return url
    }
    
    @MainActor
    func requestModelFolder() -> URL?{
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.message = "Select the folder containing the model"
        
        let response = panel.runModal()
        
        guard response == .OK, let url = panel.url else {
            return nil
        }
        return url
        
    }
    
}
