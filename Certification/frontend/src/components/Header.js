import React from 'react';

function Header({ logo, selectedSchoolDistrict, setNeedsSchoolSelection, isAuthenticated, needsSchoolSelection, user }) {
  return (
    <div className="header">
      <div 
        onClick={() => {
          console.log('School icon clicked!');
          console.log('Current state - isAuthenticated:', isAuthenticated, 'needsSchoolSelection:', needsSchoolSelection, 'user:', user);
          setNeedsSchoolSelection(true);
          console.log('Set needsSchoolSelection to true');
        }}
        className="header-logo-clickable"
        title="Change school"
      >
        <img src={logo} alt="School Assistant" className="header-icon" />
      </div>
      <h1>School Assistant</h1>
      {selectedSchoolDistrict && (
        <div 
          className="selected-school-badge"
          onClick={() => {
            setNeedsSchoolSelection(true);
          }}
          title="Click to change school"
        >
          📍 {selectedSchoolDistrict.district}
        </div>
      )}
    </div>
  );
}

export default Header;
