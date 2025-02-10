import React from "react";
import styled from "styled-components";
import { COLORS, FONTS } from "../../styles/theme";
import SelectBox from "./SelectBox";
import NumberInput from "./NumberInput";

function PromotionForm({
  showArrow = true,
  showTitle = true,
  brand = "",
  onChangeBrand = () => {},
  product = "",
  onChangeProduct = () => {},
  price = "",
  onChangePrice = () => {},
}) {
  return (
    <Container>
      {showTitle && <Title>PROMO를 통해 최적의 프로모션을 설정해보세요</Title>}
      <InputRow>
        <SelectBox
          defaultLabel="브랜드"
          items={["SKAD", "SAMSUNG", "XIAOMI", "OGOK"]}
          value={brand}
          onChange={onChangeBrand}
          showArrow={showArrow}
          width="260px"
          height="39px"
        />
        <SelectBox
          defaultLabel="상품"
          items={[
            "제네바 그래파이트",
            "크레타 실버",
            "센터링",
            "사무라이 흑연",
          ]}
          value={product}
          onChange={onChangeProduct}
          showArrow={showArrow}
          width="260px"
          height="39px"
        />
        <NumberInput
          width="260px"
          height="39px"
          placeholder="판매가"
          value={price}
          onChange={onChangePrice}
          readMode={!showArrow}
        />
      </InputRow>
    </Container>
  );
}

export default PromotionForm;

const Container = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
`;

const Title = styled.h2`
  color: ${COLORS.B1};
  ${FONTS.mainLogo};
  text-align: center;
  margin-bottom: 40px;
`;

const InputRow = styled.div`
  display: flex;
  gap: 37px;
`;
