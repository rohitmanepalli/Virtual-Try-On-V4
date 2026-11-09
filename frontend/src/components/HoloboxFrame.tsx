import React from "react";

type Props = {
  children: React.ReactNode;
};

export default function HoloboxFrame({ children }: Props) {

  return (
    <div
      style={{
        width: "100vw",
        height: "100vh",
        background: "#453333ff",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        overflow: "hidden", // Prevent scrollbars
      }}
    >
      {/* 
        Main Application Container
        Enforces 140:206 aspect ratio based on physical Holobox dimensions.
      */}
      <div
        style={{
          aspectRatio: "140 / 206",
          height: "100%", // Take up full height of the display
          maxHeight: "100%",
          backgroundColor: "#FFFFFF",
          display: "flex",
          flexDirection: "column",
          position: "relative",
          overflow: "hidden", // Clip background layers to aspect ratio frame
        }}
      >


        {/* 
          SAFE ZONE CONTAINER 
          Top Margin: 8%
          Bottom Margin: 8%
          Left Margin: 6%
          Right Margin: 6%
        */}
        <div
          style={{
            width: "100%",
            height: "100%",
            paddingTop: "8%",
            paddingBottom: "8%",
            paddingLeft: "6%",
            paddingRight: "6%",
            display: "flex",
            flexDirection: "column",
            position: "relative",
            zIndex: 2, // Sit overlay elements on top of background video
          }}
        >
          {children}
        </div>
      </div>
    </div>
  );
}