import React from 'react';
import { BookOpen } from 'lucide-react';

const BookmarksContainer = ({ bookmarks, removeBookmark, formatResponseText, BookmarkIcon }) => (
  <div className="chat-container">
    <div className="bookmarks-container" style={{ background: '#f4f7fb', padding: 24, borderRadius: 16 }}>
      {bookmarks.length === 0 ? (
        <div className="welcome-message" style={{ textAlign: 'center', padding: '48px 32px', background: '#fff', borderRadius: 14, boxShadow: '0 4px 16px rgba(25, 118, 210, 0.10)', color: '#1a237e', fontWeight: 600, fontSize: '1.15em' }}>
          <strong style={{ fontSize: '1.25em', color: '#0d1a26' }}>No bookmarks yet</strong>
          <br /><br />
          <span style={{ color: '#444', fontWeight: 500 }}>Bookmark assistant messages from the Chat tab to save them here for later reference.</span>
          <br /><br />
          <span style={{display: 'inline-flex', alignItems: 'center', gap: '6px', color: '#1976d2', fontWeight: 700}}>
            Look for the <BookmarkIcon size={20} style={{verticalAlign: 'middle', color: '#1976d2'}} /> icon on assistant messages!
          </span>
        </div>
      ) : (
        <div className="bookmarks-list">
          {bookmarks.map((bookmark, index) => (
            <div key={bookmark.bookmark_id} className="bookmark-item">
              <button
                onClick={() => removeBookmark(bookmark.bookmark_id)}
                className="bookmark-remove-button"
                title="Remove bookmark"
              >
                Remove
              </button>
              <div className="bookmark-saved-date" style={{ color: '#1a237e', fontWeight: 700, fontSize: '1em', marginBottom: 10, letterSpacing: '0.2px' }}>
                Saved on {new Date(bookmark.created_at).toLocaleString()}
              </div>
              <div className="message-content" style={{ color: '#22272e', fontSize: '1.04em', lineHeight: 1.72 }}>
                <div className="message-text">
                  {formatResponseText(bookmark.message_content)}
                </div>
                {bookmark.message_source && (
                  <div className="bookmark-source" style={{ marginTop: 14, fontSize: '0.97em', color: '#1565c0', fontWeight: 600, background: '#e3eafc', borderRadius: 6, padding: '8px 12px' }}>
                    <BookOpen size={18} style={{ display: 'inline', marginRight: '6px', verticalAlign: 'middle', color: '#1976d2' }} /> Source: {bookmark.message_source}
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  </div>
);

export default BookmarksContainer;
