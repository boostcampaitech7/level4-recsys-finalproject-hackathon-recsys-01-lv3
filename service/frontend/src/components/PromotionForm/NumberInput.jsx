import React from "react";
import styled from "styled-components";
import { COLORS, FONTS } from "../../styles/theme";

function NumberInput({
  width = "260px",
  height = "39px",
  placeholder = "",
  value = "",
  onChange = () => {},
  readMode = false,
}) {
  const handleChange = (e) => {
    if (readMode) return;
    const raw = e.target.value;
    const numericOnly = raw.replace(/\D/g, "");
    onChange(numericOnly);
  };

  const displayValue = value ? `$${value}` : "";

  return (
    <InputContainer width={width} height={height}>
      <StyledInput
        type="text"
        placeholder={placeholder}
        value={displayValue}
        onChange={handleChange}
        readOnly={readMode}
      />
    </InputContainer>
  );
}

export default NumberInput;

const InputContainer = styled.div`
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

  display: flex;
  align-items: center;
  justify-content: center;
`;

const StyledInput = styled.input`
  width: 90%;
  border: none;
  outline: none;
  background: none;
  text-align: center;

  ${FONTS.selectionText};
  color: ${COLORS.B2};

  &::placeholder {
    color: ${COLORS.B2};
    opacity: 0.5;
  }
`;
