import React from "react";
import { useNavigate } from "react-router-dom";
import styled from "styled-components";
import { COLORS, FONTS } from "../styles/theme";
import PromotionForm from "../components/PromotionForm/PromotionForm";

function SimilarSearchResult() {
  const navigate = useNavigate();

  const item1 = {
    brand: "SKAD",
    product: "제네바 그래파이트",
    price: "530",
  };
  const item2 = {
    brand: "K&K",
    product: "105 블랙,실버",
    price: "754",
  };
  const item3 = {
    brand: "SKAD",
    product: "크레타 실버",
    price: "296",
  };
  const item4 = {
    brand: "SKAD",
    product: "디스크 블랙",
    price: "529",
  };

  const handlePromoStart = (itemData) => {
    navigate("/result1", { state: { ...itemData } });
  };

  return (
    <Container>
      <TopText>
        아직 판매하고 있는 상품이 없으신가요?
        <br />
        유사 상품으로 최적가격과 타겟유저를 설정해드립니다
      </TopText>

      <Spacing height="55px" />
      <SimilarItemTitle>유사 상품1</SimilarItemTitle>
      <Spacing height="10px" />
      <Line />
      <Spacing height="35px" />
      <PromotionForm
        showArrow={false}
        showTitle={false}
        brand={item1.brand}
        product={item1.product}
        price={item1.price}
      />
      <Spacing height="35px" />
      <PromoStartButton onClick={() => handlePromoStart(item1)}>
        PROMO START
      </PromoStartButton>

      <Spacing height="60px" />
      <SimilarItemTitle>유사 상품2</SimilarItemTitle>
      <Spacing height="10px" />
      <Line />
      <Spacing height="35px" />
      <PromotionForm
        showArrow={false}
        showTitle={false}
        brand={item2.brand}
        product={item2.product}
        price={item2.price}
      />
      <Spacing height="35px" />
      <PromoStartButton onClick={() => handlePromoStart(item2)}>
        PROMO START
      </PromoStartButton>

      <Spacing height="60px" />
      <SimilarItemTitle>유사 상품3</SimilarItemTitle>
      <Spacing height="10px" />
      <Line />
      <Spacing height="35px" />
      <PromotionForm
        showArrow={false}
        showTitle={false}
        brand={item3.brand}
        product={item3.product}
        price={item3.price}
      />
      <Spacing height="35px" />
      <PromoStartButton onClick={() => handlePromoStart(item3)}>
        PROMO START
      </PromoStartButton>

      <Spacing height="60px" />
      <SimilarItemTitle>유사 상품4</SimilarItemTitle>
      <Spacing height="10px" />
      <Line />
      <Spacing height="35px" />
      <PromotionForm
        showArrow={false}
        showTitle={false}
        brand={item4.brand}
        product={item4.product}
        price={item4.price}
      />
      <Spacing height="35px" />
      <PromoStartButton onClick={() => handlePromoStart(item4)}>
        PROMO START
      </PromoStartButton>

      <Spacing height="200px" />
    </Container>
  );
}

export default SimilarSearchResult;

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
`;

const Spacing = styled.div`
  height: ${(props) => props.height || "0px"};
`;

const SimilarItemTitle = styled.h3`
  ${FONTS.bigProductTitle};
  color: ${COLORS.B1};
  margin: 0;
`;

const Line = styled.hr`
  width: 994px;
  border: none;
  border-top: 1px solid ${COLORS.G1};
`;

const PromoStartButton = styled.button`
  width: 180px;
  height: 44px;
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
