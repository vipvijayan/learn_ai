import React from 'react';
import { BookOpen } from 'lucide-react';

const BookmarksContainer = ({ bookmarks, removeBookmark, formatResponseText, BookmarkIcon }) => (
  <div className="chat-container">
    <div className="bookmarks-container">
      {bookmarks.length === 0 ? (
        <div className="welcome-message" style={{ textAlign: 'center', padding: '40px' }}>
          <strong>No bookmarks yet</strong>
          <br /><br />
          Bookmark assistant messages from the Chat tab to save them here for later reference.
          <br /><br />
          <span style={{display: 'inline-flex', alignItems: 'center', gap: '4px'}}>
            Look for the <BookmarkIcon size={16} style={{verticalAlign: 'middle'}} /> icon on assistant messages!
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
              <div className="bookmark-saved-date">
                Saved on {new Date(bookmark.created_at).toLocaleString()}
              </div>
              <div className="message-content">
                <div className="message-text">
                  {formatResponseText(bookmark.message_content)}
                </div>
                {bookmark.message_source && (
                  <div className="bookmark-source">
                    <BookOpen size={16} style={{ display: 'inline', marginRight: '4px', verticalAlign: 'middle' }} /> Source: {bookmark.message_source}
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
