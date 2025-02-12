import React, { useState } from "react";
import styled from "styled-components";
import { useNavigate } from "react-router-dom";
import PromotionForm from "../components/PromotionForm/PromotionForm";
import PromoButton from "../components/PromoButton";

function Main() {
  const navigate = useNavigate();

  const [brand, setBrand] = useState("");
  const [product, setProduct] = useState("");
  const [price, setPrice] = useState("");

  const handleStartClick = () => {
    navigate("/result1", {
      state: {
        brand,
        product,
        price,
      },
    });
  };

  return (
    <MainContainer>
      <PromotionForm
        showArrow={true}
        brand={brand}
        onChangeBrand={setBrand}
        product={product}
        onChangeProduct={setProduct}
        price={price}
        onChangePrice={setPrice}
      />

      <ButtonWrapper>
        <PromoButton label="PROMO START" onClick={handleStartClick} />
      </ButtonWrapper>
    </MainContainer>
  );
}

export default Main;

const MainContainer = styled.div`
  margin-top: 231px;
  display: flex;
  flex-direction: column;
  align-items: center;
`;
const ButtonWrapper = styled.div`
  margin-top: 50px;
`;
