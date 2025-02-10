import React from "react";
import { useLocation, useNavigate } from "react-router-dom";
import styled from "styled-components";

import PromotionForm from "../components/PromotionForm/PromotionForm";
import OptimalPromotionBox from "../components/OptimalPromotionBox";
import DiscountControlBox from "../components/DiscountControlBox";
import GraphAndBoxes from "../components/GraphAndBoxes";
import PromoButton from "../components/PromoButton";
import FileDownloadSection from "../components/FileDownloadSection";

function Result2() {
  const navigate = useNavigate();
  const { state } = useLocation() || {};
  const {
    brand = "",
    product = "",
    price = "",
    discountRate = "",
    peopleCount = "",
  } = state || {};

  const handleGoToResult1 = () => {
    navigate("/result1");
  };

  return (
    <Result2Container>
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
        peopleCount={peopleCount}
        alwaysG1
      />
      <SpacingBlock marginTop="25px" />
      <FileDownloadSection />
      <SpacingBlock marginTop="86px" />
      <GraphAndBoxes />
      <RePromoButtonWrapper>
        <PromoButton
          label="다른 할인율, 인원수 조정해보기"
          buttonWidth="300px"
          onClick={handleGoToResult1}
        />
      </RePromoButtonWrapper>
    </Result2Container>
  );
}

export default Result2;

const Result2Container = styled.div`
  margin-top: 231px;
  margin-bottom: 100px;
`;

const SpacingBlock = styled.div`
  margin-top: ${(props) => props.marginTop || "0px"};
`;

const RePromoButtonWrapper = styled.div`
  margin-top: 50px;
  display: flex;
  justify-content: center;
`;
