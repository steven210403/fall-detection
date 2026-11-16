import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { FallHistory } from "../src/components/FallHistory";
import { fallEvent } from "./fixtures";

describe("FallHistory", () => {
  it("says when nothing is stored", () => {
    render(<FallHistory events={[]} hasMore={false} onLoadMore={vi.fn()} />);
    expect(screen.getByRole("status")).toHaveTextContent("Belum ada kejadian jatuh");
  });

  it("says it is reading before the first page, not that nothing is stored", () => {
    render(<FallHistory events={[]} hasMore={false} onLoadMore={vi.fn()} loading />);
    expect(screen.getByRole("status")).toHaveTextContent("Membaca riwayat");
  });

  it("shows its own read error with a retry", () => {
    const onRetry = vi.fn();
    render(
      <FallHistory events={[]} hasMore={false} onLoadMore={vi.fn()} error="server tidak menjawab" onRetry={onRetry} />,
    );
    expect(screen.getByRole("alert")).toHaveTextContent("Riwayat tidak dapat dibaca: server tidak menjawab");
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Baca ulang riwayat" }));
    expect(onRetry).toHaveBeenCalled();
  });

  it("marks an open event as active", () => {
    render(<FallHistory events={[fallEvent()]} hasMore={false} onLoadMore={vi.fn()} />);
    expect(screen.getByRole("listitem")).toHaveTextContent("aktif");
    expect(screen.getByRole("listitem")).toHaveTextContent("respons 3.0 dtk");
  });

  it("explains why a closed event ended", () => {
    render(
      <FallHistory
        events={[fallEvent({ active: false, ended_utc: 1_700_000_013.5, end_reason: "recovered" })]}
        hasMore={false}
        onLoadMore={vi.fn()}
      />,
    );
    expect(screen.getByRole("listitem")).toHaveTextContent("bangkit kembali");
    expect(screen.getByRole("listitem")).toHaveTextContent("durasi 10.0 dtk");
  });

  it("loads older history", () => {
    const onLoadMore = vi.fn();
    render(<FallHistory events={[fallEvent()]} hasMore onLoadMore={onLoadMore} />);
    fireEvent.click(screen.getByRole("button", { name: "Muat lebih lama" }));
    expect(onLoadMore).toHaveBeenCalled();
  });
});
