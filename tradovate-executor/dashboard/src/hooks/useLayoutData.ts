import { useOutletContext } from "react-router-dom";
import type { WSData } from "../types";

export function useLayoutData(): WSData {
  return useOutletContext() as WSData;
}
