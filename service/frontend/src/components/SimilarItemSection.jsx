import React, { useState } from "react";
import styled from "styled-components";
import PromotionForm from "../components/PromotionForm/PromotionForm";

function SimilarItemSection({ title, onPromoStart }) {
  const [brand, setBrand] = useState("");
  const [product, setProduct] = useState("");
  const [price, setPrice] = useState("");

  const handlePromoStartClick = () => {
    onPromoStart({ brand, product, price });
  };

  return (
    <SectionContainer>
      <SimilarItemTitle>{title}</SimilarItemTitle>
      <Spacing height="10px" />
      <Line />
      <Spacing height="35px" />
      <PromotionForm
        showArrow={true}
        showTitle={false}
        brand={brand}
        onChangeBrand={setBrand}
        product={product}
        onChangeProduct={setProduct}
        price={price}
        onChangePrice={setPrice}
      />
      <Spacing height="35px" />
      <PromoStartButton onClick={handlePromoStartClick}>
        PROMO START
      </PromoStartButton>
    </SectionContainer>
  );
}

export default SimilarItemSection;

const SectionContainer = styled.div`
  width: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
`;

const SimilarItemTitle = styled.h3`
  margin: 0;
`;

const Spacing = styled.div`
  height: ${(props) => props.height || "0px"};
`;

const Line = styled.hr`
  width: 994px;
  border: none;
  border-top: 1px solid #ccc;
`;

const PromoStartButton = styled.button`
  width: 180px;
  height: 44px;
  border: none;
  border-radius: 50px;
`;
