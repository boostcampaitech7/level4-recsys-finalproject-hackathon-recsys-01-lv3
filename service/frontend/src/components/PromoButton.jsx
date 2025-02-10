import React from "react";
import styled from "styled-components";
import { COLORS, FONTS } from "../styles/theme";

function PromoButton({ label = "PROMO START", onClick, buttonWidth }) {
  return (
    <StyledButton onClick={onClick} buttonWidth={buttonWidth}>
      {label}
    </StyledButton>
  );
}

export default PromoButton;

const StyledButton = styled.button`
  height: 44px;
  border: none;
  border-radius: 50px;
  cursor: pointer;
  background-color: ${COLORS.B2};
  color: ${COLORS.W1};
  ${FONTS.buttonText};
  width: ${(props) => props.buttonWidth || "auto"};
  padding: 0 32px;
  box-shadow: 1px 4px 4px rgba(194, 194, 194, 0.25),
    inset -1px -3px 5px rgba(255, 255, 255, 0.25);

  &:hover {
    opacity: 0.9;
  }
`;
