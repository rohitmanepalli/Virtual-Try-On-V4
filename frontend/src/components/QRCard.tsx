/**
 * QRCard component: renders a white rounded card
 * containing the QR code image from the backend API response.
 * Falls back to a placeholder if no QR data is provided.
 */

type QRCardProps = {
  qrCodeSrc?: string | null;
};

export default function QRCard({ qrCodeSrc }: QRCardProps) {
  return (
    <div
      style={{
        width: "12.0vh",
        height: "12.0vh",
        background: "transparent",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
      }}
    >
      {qrCodeSrc ? (
        <img
          src={qrCodeSrc}
          alt="Scan QR Code to download"
          style={{
            width: "100%",
            height: "100%",
            objectFit: "contain",
            borderRadius: "0.5vh",
          }}
        />
      ) : (
        /* Fallback placeholder - shown only if backend didn't return a QR code */
        <div
          style={{
            width: "100%",
            height: "100%",
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            color: "#999",
            fontFamily: "'Poppins', sans-serif",
            fontSize: "0.88vh",
            textAlign: "center",
          }}
        >
          QR Code Loading...
        </div>
      )}
    </div>
  );
}
