import React from "react";
import styled from "styled-components";
import { COLORS, FONTS } from "../styles/theme";

function GraphAndBoxes() {
  return (
    <Container>
      <TopSection>
        <PlotlyIframe
          src="//plotly.com/~ygw97602/215.embed"
          width="700"
          height="650"
          frameBorder="0"
          scrolling="no"
          title="Plotly3D"
        />
        <RightBoxes>
          <BoxTitleBox>
            <BoxTitle>추가 예상 매출</BoxTitle>
          </BoxTitleBox>
          <BoxValue style={{ marginTop: "10px" }}>$56107.8</BoxValue>
          <Spacer height="20px" />
          <BoxTitleBox>
            <BoxTitle>최적매출과 차이</BoxTitle>
          </BoxTitleBox>
          <BoxValue style={{ marginTop: "10px" }}>$-144,416.57</BoxValue>
        </RightBoxes>
      </TopSection>
      <BottomBox>
        <BottomText>
          현재 설정된 할인율(20%)은 적절한 수준이지만, 추천 대상 유저
          수(700명)가 다소 부족합니다. 추천 대상을 9911명으로 늘리면 판매 상승
          효과가 극대화될 것입니다. 또한, 할인율을 약간 낮춰(15%) 추가적인 구매
          전환을 유도하는 것도 고려해보세요.
        </BottomText>
      </BottomBox>
    </Container>
  );
}

export default GraphAndBoxes;

const Container = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
`;

const TopSection = styled.div`
  display: flex;
  align-items: center;
  gap: 50px;
`;

const PlotlyIframe = styled.iframe`
  border: none;
`;

const RightBoxes = styled.div`
  display: flex;
  flex-direction: column;
`;

const BoxTitleBox = styled.div`
  width: 187px;
  height: 50px;
  background-color: ${COLORS.G1};
  border-radius: 50px;
  display: flex;
  align-items: center;
  justify-content: center;
`;

const BoxTitle = styled.div`
  ${FONTS.buttonText};
  color: ${COLORS.B1};
`;

const BoxValue = styled.div`
  ${FONTS.promo};
  color: ${COLORS.B2};
  text-align: center;
`;

const Spacer = styled.div`
  height: ${(props) => props.height || "10px"};
`;

const BottomBox = styled.div`
  width: 790px;
  background-color: ${COLORS.G1};
  border-radius: 50px;
  margin-top: 50px;
  padding: 20px 30px;
`;

const BottomText = styled.div`
  ${FONTS.buttonText};
  color: ${COLORS.B1};
  line-height: 1.4;
`;
