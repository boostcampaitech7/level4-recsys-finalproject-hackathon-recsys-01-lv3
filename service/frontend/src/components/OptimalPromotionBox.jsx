import React from "react";
import styled from "styled-components";
import { COLORS, FONTS } from "../styles/theme";

function OptimalPromotionBox({ product = "" }) {
  return (
    <Container>
      <Title>{product || "PRODUCT"}의 최적 프로모션</Title>
      <Line />
      <Row>
        <Column>
          <Label>최적 프로모션 대상 인원 수</Label>
          <Value>9911명</Value>
        </Column>
        <Column>
          <Label>최적 할인율</Label>
          <Value>15%</Value>
        </Column>
        <Column>
          <Label>최대 추가 예상 매출</Label>
          <Value>$200524.37</Value>
        </Column>
      </Row>
    </Container>
  );
}

export default OptimalPromotionBox;

const Container = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
`;

const Title = styled.h2`
  color: ${COLORS.B1};
  ${FONTS.mainLogo};
  text-align: center;
`;

const Line = styled.div`
  margin-bottom: 41px;
  width: 994px;
  height: 3px;
  background-color: ${COLORS.G1};
`;

const Row = styled.div`
  display: flex;
  gap: 63px;
`;

const Column = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
`;

const Label = styled.div`
  color: ${COLORS.B1};
  ${FONTS.buttonText};
`;

const Value = styled.div`
  margin-top: 9px;
  color: ${COLORS.B1};
  ${FONTS.promo};
`;
