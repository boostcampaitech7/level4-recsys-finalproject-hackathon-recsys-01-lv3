import React from "react";
import styled from "styled-components";
import { useNavigate } from "react-router-dom";
import { COLORS, FONTS } from "../styles/theme";

function Header() {
  const navigate = useNavigate();

  return (
    <HeaderContainer>
      <Logo onClick={() => navigate("/")}>PROMO</Logo>
      <SearchButton
        onClick={() => {
          navigate("/similarsearch");
        }}
      >
        유사 상품 검색
      </SearchButton>
    </HeaderContainer>
  );
}

export default Header;

const HeaderContainer = styled.header`
  width: 100%;
  height: 90px;
  background-color: ${COLORS.B1};
  display: flex;
  align-items: center;
  justify-content: space-between;
`;

const Logo = styled.div`
  margin-left: 44px;
  color: ${COLORS.W1};
  cursor: pointer;
  ${FONTS.mainLogo};

  &:hover {
    opacity: 0.8;
  }
`;

const SearchButton = styled.button`
  margin-right: 44px;
  background-color: ${COLORS.B1};
  color: ${COLORS.W1};
  border: none;
  border-radius: 50px;
  padding: 8px 24px;
  cursor: pointer;
  ${FONTS.buttonText};

  &:hover {
    background-color: ${COLORS.W1};
    color: ${COLORS.B1};
  }
`;
