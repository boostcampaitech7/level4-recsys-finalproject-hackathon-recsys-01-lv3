import React from "react";
import styled from "styled-components";
import { COLORS, FONTS } from "../styles/theme";

function FileDownloadSection() {
  const handleDownloadClick = () => {
    alert("파일이 다운되었습니다");
  };

  return (
    <Container>
      <FileText>프로모션 대상 파일 다운로드</FileText>
      <DownloadButton onClick={handleDownloadClick}>
        DOWNLOAD FILE
      </DownloadButton>
    </Container>
  );
}

export default FileDownloadSection;

const Container = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 16px;
`;

const FileText = styled.div`
  ${FONTS.buttonText};
  color: ${COLORS.B1};
`;

const DownloadButton = styled.button`
  width: 180px;
  height: 44px;
  border: none;
  border-radius: 50px;
  background-color: ${COLORS.B1};
  color: ${COLORS.W1};
  cursor: pointer;
  ${FONTS.buttonText};

  &:hover {
    opacity: 0.9;
  }
`;
