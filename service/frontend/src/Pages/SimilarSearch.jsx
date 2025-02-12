import React from "react";
import styled from "styled-components";
import { useNavigate } from "react-router-dom";
import { COLORS, FONTS } from "../styles/theme";

function SimilarSearch() {
  const navigate = useNavigate();

  const handleSimilarProductClick = () => {
    navigate("/similarsearchresult");
  };

  return (
    <Container>
      <TopText>
        아직 판매하고 있는 상품이 없으신가요?
        <br />
        유사 상품으로 최적가격과 타겟유저를 설정해드립니다
      </TopText>

      <DescriptionBox>
        <DescriptionTextarea placeholder="상품 설명을 입력해주세요." />
      </DescriptionBox>

      <SimilarButton onClick={handleSimilarProductClick}>
        SIMILAR PRODUCT
      </SimilarButton>
    </Container>
  );
}

export default SimilarSearch;

const Container = styled.div`
  margin-top: 231px;
  display: flex;
  flex-direction: column;
  align-items: center;
`;

const TopText = styled.h2`
  ${FONTS.mainLogo};
  color: ${COLORS.B1};
  text-align: center;
  line-height: 1.4;
  margin-bottom: 36px;
`;

const DescriptionBox = styled.div`
  width: 583px;
  height: 200px;
  border: 1px solid ${COLORS.B1};
  border-radius: 20px;
  box-shadow: 0 1px 4px rgba(2, 77, 139, 0.2),
    inset -1px -3px 5.8px rgba(2, 77, 139, 0.15);
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 50px;
`;

const DescriptionTextarea = styled.textarea`
  ${FONTS.buttonText};
  color: ${COLORS.B2};
  width: 100%;
  height: 100%;
  border: none;
  outline: none;
  resize: none;
  background: transparent;
  text-align: center;
  vertical-align: middle;
  line-height: 1.4;
  overflow: auto;
`;

const SimilarButton = styled.button`
  width: 230px;
  height: 41px;
  border: none;
  border-radius: 50px;
  background-color: ${COLORS.B2};
  color: ${COLORS.W1};
  cursor: pointer;
  ${FONTS.buttonText};

  &:hover {
    opacity: 0.9;
  }
`;
