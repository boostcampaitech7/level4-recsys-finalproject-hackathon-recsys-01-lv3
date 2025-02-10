import React from "react";
import styled from "styled-components";
import { COLORS, FONTS } from "../styles/theme";

function DiscountControlBox({
  discountRate = "",
  onChangeDiscountRate = () => {},
  peopleCount = "",
  onChangePeopleCount = () => {},
  alwaysG1 = false,
}) {
  const handleRateClick = (rate) => {
    onChangeDiscountRate(rate);
  };

  return (
    <Container>
      <Title>할인율, 인원수 조정해보기</Title>
      <Line />

      <MarginBox height="30px" />

      <RateRow>
        <RateLabel>할인율</RateLabel>
        <RateButtonsWrapper>
          {["05%", "10%", "15%", "20%", "25%", "30%"].map((rate) => (
            <RateButton
              key={rate}
              onClick={() => handleRateClick(rate)}
              isActive={discountRate === rate}
            >
              {rate}
            </RateButton>
          ))}
        </RateButtonsWrapper>
      </RateRow>

      <MarginBox height="25px" />

      <PeopleRow>
        <PeopleLabel>프로모션 대상 인원수</PeopleLabel>
        <PeopleInput
          type="text"
          placeholder="00명"
          value={peopleCount}
          onChange={(e) => onChangePeopleCount(e.target.value)}
          hasValue={peopleCount !== ""}
          alwaysG1={alwaysG1}
        />
      </PeopleRow>
    </Container>
  );
}

export default DiscountControlBox;

const Container = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
`;

const Title = styled.h2`
  margin: 0;
  color: ${COLORS.B1};
  ${FONTS.mainLogo};
  text-align: center;
`;

const Line = styled.div`
  margin-top: 20px;
  width: 994px;
  height: 3px;
  background-color: ${COLORS.G1};
`;

const MarginBox = styled.div`
  width: 100%;
  height: ${(props) => props.height || "20px"};
`;

const RateRow = styled.div`
  display: flex;
  align-items: center;
  gap: 46px;
`;

const RateLabel = styled.div`
  ${FONTS.buttonText};
  color: ${COLORS.B1};
`;

const RateButtonsWrapper = styled.div`
  display: flex;
  gap: 10px;
`;

const RateButton = styled.button`
  width: 100px;
  height: 44px;
  border: 1px solid ${COLORS.G1};
  border-radius: 50px;
  background-color: #fff;
  cursor: pointer;
  ${FONTS.buttonText};
  color: ${COLORS.B1};

  &:hover {
    background-color: ${COLORS.G1};
  }

  ${(props) =>
    props.isActive &&
    `
      background-color: ${COLORS.G1};
    `}
`;

const PeopleRow = styled.div`
  display: flex;
  align-items: center;
  gap: 46px;
`;

const PeopleLabel = styled.div`
  ${FONTS.buttonText};
  color: ${COLORS.B1};
`;

const PeopleInput = styled.input`
  width: 100px;
  height: 44px;
  border: 1px solid ${COLORS.G1};
  border-radius: 50px;
  text-align: center;
  ${FONTS.buttonText};
  color: ${COLORS.B1};
  cursor: pointer;

  background-color: ${(props) =>
    props.alwaysG1 ? COLORS.G1 : props.hasValue ? COLORS.G1 : "#fff"};

  &:hover {
    background-color: ${COLORS.G1};
  }
`;
