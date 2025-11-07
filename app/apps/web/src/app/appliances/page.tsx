"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Loader2, Power, ArrowLeft, ArrowRight } from "lucide-react";
import { useMe } from "@/lib/hooks";
import axios from "axios";

interface UserProfile {
	isParalyzed: boolean;
}

interface PinState {
	d0: boolean;
	d5: boolean;
	d6: boolean;
	d7: boolean;
}

const APPLIANCE_API_URL = "http://localhost:8000/api/appliances/control";
const APPLIANCE_PASSWORD = "appliances123";

export default function AppliancesPage() {
	const router = useRouter();
	const { data: user, isLoading, error } = useMe();
	const isParalyzedUser = Boolean(
		(user as UserProfile | undefined)?.isParalyzed ?? (user as any)?.is_paralysed
	);

	const [pinStates, setPinStates] = useState<PinState>({
		d0: false,
		d5: false,
		d6: false,
		d7: false,
	});

	const [loadingPins, setLoadingPins] = useState<Set<string>>(new Set());
	const [controlError, setControlError] = useState<string | null>(null);

	if (isLoading) {
		return (
			<div className="h-screen w-screen flex items-center justify-center bg-white dark:bg-gray-950">
				<Loader2 className="w-10 h-10 animate-spin text-blue-500" />
			</div>
		);
	}

	if (error) {
		return (
			<div className="h-screen w-screen flex items-center justify-center bg-white dark:bg-gray-950">
				<div className="text-center space-y-2">
					<p className="text-red-500 font-medium">Unable to load your profile</p>
					<p className="text-sm text-gray-600 dark:text-gray-400">
						{error instanceof Error ? error.message : "Unknown error"}
					</p>
				</div>
			</div>
		);
	}

	const handlePinToggle = async (pin: keyof PinState) => {
		const newState = !pinStates[pin];
		
		// Optimistic update
		setPinStates((prev) => ({ ...prev, [pin]: newState }));
		setLoadingPins((prev) => new Set(prev).add(pin));
		setControlError(null);

		try {
			const response = await axios.post(APPLIANCE_API_URL, {
				pins: {
					[pin]: newState ? "on" : "off",
				},
				password: APPLIANCE_PASSWORD,
			});

			if (!response.data.success) {
				throw new Error(response.data.message || "Failed to control appliance");
			}
		} catch (err) {
			console.error(`Failed to toggle ${pin}:`, err);
			// Revert on error
			setPinStates((prev) => ({ ...prev, [pin]: !newState }));
			setControlError(
				err instanceof Error
					? err.message
					: "Unable to control appliance right now."
			);
		} finally {
			setLoadingPins((prev) => {
				const updated = new Set(prev);
				updated.delete(pin);
				return updated;
			});
		}
	};

	const handleBack = () => {
		router.push("/chat");
	};

	const handleNext = () => {
		// You can navigate to another page if needed
		router.push("/chat");
	};

	const appliances = [
		{ pin: "d0" as keyof PinState, label: "Appliance 1", color: "bg-blue-500", ring: "ring-blue-300" },
		{ pin: "d5" as keyof PinState, label: "Appliance 2", color: "bg-green-500", ring: "ring-green-300" },
		{ pin: "d6" as keyof PinState, label: "Appliance 3", color: "bg-purple-500", ring: "ring-purple-300" },
		{ pin: "d7" as keyof PinState, label: "Appliance 4", color: "bg-orange-500", ring: "ring-orange-300" },
	];

	const renderApplianceCard = (appliance: typeof appliances[0], index: number) => {
		const isOn = pinStates[appliance.pin];
		const isLoading = loadingPins.has(appliance.pin);

		return (
			<button
				key={appliance.pin}
				onClick={() => handlePinToggle(appliance.pin)}
				disabled={isLoading}
				className="relative flex h-full w-full cursor-pointer flex-col items-center justify-center border border-gray-200 dark:border-gray-800 group"
			>
				<div
					className={`absolute inset-0 m-5 ${appliance.color} rounded-2xl border-4 border-transparent transition-all duration-300 group-hover:scale-[1.02] group-hover:shadow-xl group-hover:${appliance.ring} ${
						isOn ? "scale-105 ring-4 ring-white" : "opacity-60"
					} flex flex-col items-center justify-center p-6 text-white`}
				>
					<div className="absolute top-4 left-4 w-12 h-12 rounded-full bg-white bg-opacity-30 flex items-center justify-center text-3xl font-black">
						{index + 1}
					</div>

					{isLoading ? (
						<div className="absolute inset-0 flex items-center justify-center">
							<Loader2 className="w-16 h-16 animate-spin" />
						</div>
					) : (
						<>
							<Power className={`w-20 h-20 mb-4 ${isOn ? "animate-pulse" : ""}`} />
							<div className="text-2xl font-bold mb-2">{appliance.label}</div>
							<div className="text-lg font-medium">
								{isOn ? "ON" : "OFF"}
							</div>
						</>
					)}
				</div>
			</button>
		);
	};

	return (
		<div className="h-screen w-screen bg-white dark:bg-gray-950 overflow-hidden fixed inset-0">
			<div className="h-16 bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-800 flex items-center px-4 gap-4">
				<button
					onClick={handleBack}
					className="w-10 h-10 flex items-center justify-center hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors"
				>
					<ArrowLeft className="w-6 h-6" />
				</button>

				<div className="flex items-center gap-3 flex-1">
					<div>
						<h2 className="text-lg font-bold text-gray-900 dark:text-white">
							Appliance Control
						</h2>
						<p className="text-sm text-gray-500">Tap to turn appliances on/off</p>
					</div>
				</div>
			</div>

			{controlError && (
				<div className="px-6 py-3 bg-red-50 dark:bg-red-900/20 border-b border-red-200 dark:border-red-800">
					<p className="text-sm text-red-600 dark:text-red-400">{controlError}</p>
				</div>
			)}

			<div className="grid grid-cols-3 grid-rows-2 h-[calc(100vh-4rem)] w-full gap-0">
				<button
					onClick={handleBack}
					className="relative flex flex-col items-center justify-center border border-gray-200 dark:border-gray-800 group"
				>
					<div className="absolute inset-0 m-5 bg-gray-600 hover:bg-gray-700 group-hover:scale-[1.02] rounded-2xl flex flex-col items-center justify-center transition-all duration-300 group-hover:shadow-xl text-white">
						<ArrowLeft className="w-16 h-16 mb-4 group-hover:-translate-x-2 transition-transform duration-300" />
						<div className="text-2xl font-bold">Back</div>
					</div>
				</button>

				{appliances.map((appliance, index) => renderApplianceCard(appliance, index))}

				<button
					onClick={handleNext}
					className="relative flex flex-col items-center justify-center border border-gray-200 dark:border-gray-800 group"
				>
					<div className="absolute inset-0 m-5 bg-gray-600 hover:bg-gray-700 group-hover:scale-[1.02] rounded-2xl flex flex-col items-center justify-center transition-all duration-300 group-hover:shadow-xl text-white">
						<ArrowRight className="w-16 h-16 mb-4 group-hover:translate-x-2 transition-transform duration-300" />
						<div className="text-2xl font-bold">Next</div>
					</div>
				</button>
			</div>
		</div>
	);
}
