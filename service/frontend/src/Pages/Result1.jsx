import React, { useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import styled from "styled-components";

import PromotionForm from "../components/PromotionForm/PromotionForm";
import OptimalPromotionBox from "../components/OptimalPromotionBox";
import DiscountControlBox from "../components/DiscountControlBox";
import PromoButton from "../components/PromoButton";

function Result1() {
  const navigate = useNavigate();
  const { state } = useLocation() || {};
  const { brand = "", product = "", price = "" } = state || {};

  const [discountRate, setDiscountRate] = useState("");
  const [peopleCount, setPeopleCount] = useState("");

  const handleRePromoClick = () => {
    navigate("/result2", {
      state: {
        brand,
        product,
        price,
        discountRate,
        peopleCount,
      },
    });
  };

  return (
    <ResultContainer>
      <PromotionForm
        showArrow={false}
        brand={brand}
        product={product}
        price={price}
      />
      <SpacingBlock marginTop="150px" />
      <OptimalPromotionBox product={product} />
      <SpacingBlock marginTop="150px" />
      <DiscountControlBox
        discountRate={discountRate}
        onChangeDiscountRate={setDiscountRate}
        peopleCount={peopleCount}
        onChangePeopleCount={setPeopleCount}
      />
      <RePromoButtonWrapper>
        <PromoButton label="RE-PROMO" onClick={handleRePromoClick} />
      </RePromoButtonWrapper>
    </ResultContainer>
  );
}

export default Result1;

const ResultContainer = styled.div`
  margin-top: 231px;
  margin-bottom: 500px;
`;

export const SpacingBlock = styled.div`
  margin-top: ${(props) => props.marginTop || "0px"};
`;

const RePromoButtonWrapper = styled.div`
  margin-top: 50px;
  display: flex;
  justify-content: center;
`;
