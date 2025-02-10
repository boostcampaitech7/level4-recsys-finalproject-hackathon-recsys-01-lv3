import React, { useState } from "react";
import styled from "styled-components";
import { COLORS, FONTS } from "../../styles/theme";
import arrowDown from "../../assets/icons/arrow-down.svg";

function SelectBox({
  items = [],
  value = "",
  onChange = () => {},
  defaultLabel = "선택",
  showArrow = true,
  width = "260px",
  height = "39px",
}) {
  const [isOpen, setIsOpen] = useState(false);

  const handleToggleDropdown = () => {
    if (!showArrow) return;
    setIsOpen(!isOpen);
  };

  const handleSelect = (selected) => {
    onChange(selected);
    setIsOpen(false);
  };

  return (
    <SelectContainer width={width} height={height}>
      <SelectHeader onClick={handleToggleDropdown}>
        <LabelText>{value || defaultLabel}</LabelText>
        {showArrow && <ArrowIcon src={arrowDown} alt="arrow-down" />}
      </SelectHeader>

      {showArrow && isOpen && (
        <DropdownMenu>
          {items.map((item) => (
            <DropdownItem key={item} onClick={() => handleSelect(item)}>
              {item}
            </DropdownItem>
          ))}
        </DropdownMenu>
      )}
    </SelectContainer>
  );
}

export default SelectBox;

const SelectContainer = styled.div`
  position: relative;
  width: ${(props) => props.width};
  height: ${(props) => props.height};
  border: 1px solid ${COLORS.B1};
  border-radius: 50px;
  background-color: #fff;
  box-shadow: 0 1px 4px rgba(194, 194, 194, 0.2),
    inset -1px -3px 5.8px rgba(194, 194, 194, 0.15);
  &:hover {
    border-color: ${COLORS.G1};
  }
`;

const SelectHeader = styled.div`
  width: 100%;
  height: 100%;
  border-radius: 50px;
  padding: 0 40px;
  box-sizing: border-box;
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
  cursor: pointer;
`;

const LabelText = styled.span`
  color: ${COLORS.B2};
  ${FONTS.selectionText}; // or buttonText, as needed
  text-align: center;
`;

const ArrowIcon = styled.img`
  position: absolute;
  right: 9px;
  width: 10px;
  height: 15px;
`;

const DropdownMenu = styled.div`
  position: absolute;
  top: 100%;
  left: 0;
  width: 100%;
  background-color: #fff;
  border: 1px solid ${COLORS.B1};
  border-radius: 10px;
  margin-top: 4px;
  box-shadow: 0 1px 4px rgba(194, 194, 194, 0.2),
    inset -1px -3px 5.8px rgba(194, 194, 194, 0.15);
  overflow: hidden;
`;

const DropdownItem = styled.div`
  padding: 8px 16px;
  color: ${COLORS.B1};
  text-align: center;
  cursor: pointer;
  &:hover {
    background-color: #f1f1f1;
  }
`;
