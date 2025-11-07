"use client";

import { useRouter } from "next/navigation";
import { MessageSquare, Mic, AlertCircle, Home, Eye } from "lucide-react";

export default function HomePage() {
	const router = useRouter();

	const menuItems = [
		{
			id: "chat",
			label: "Chat",
			icon: MessageSquare,
			color: "from-blue-500 to-blue-600",
			path: "/chat",
		},
		{
			id: "talk",
			label: "Talk",
			icon: Mic,
			color: "from-purple-500 to-purple-600",
			path: "/talk",
		},
		{
			id: "sos",
			label: "SOS",
			icon: AlertCircle,
			color: "from-red-500 to-red-600",
			path: null,
		},
		{
			id: "appliance",
			label: "Appliance Control",
			icon: Home,
			color: "from-green-500 to-green-600",
			path: "/appliances",
		},
		{
			id: "connect",
			label: "Eye Tracker",
			icon: Eye,
			color: "from-cyan-500 to-blue-500",
			path: "/connect",
		},
	];

	const totalSlots = 6;
	const emptySlots = Math.max(0, totalSlots - menuItems.length);

	const handleCardClick = (path: string | null) => {
		if (path) {
			router.push(path as any);
		}
	};

	return (
		<div className="h-screen w-screen bg-white dark:bg-gray-950 overflow-hidden fixed inset-0">
			<div className="grid grid-cols-3 grid-rows-2 h-full w-full gap-0">
				{menuItems.map((item, index) => {
					const Icon = item.icon;
					return (
						<div
							key={item.id}
							onClick={() => handleCardClick(item.path)}
							data-gaze-activate={item.path ? "card" : undefined}
							className={`relative flex flex-col items-center justify-center border border-gray-200 dark:border-gray-800 group ${
								item.path ? "cursor-pointer" : "cursor-default"
							}`}
						>
							<div
								className={`absolute inset-0 m-5 bg-white dark:bg-gray-900 rounded-2xl border-4 border-transparent ${
									item.path
										? "group-hover:border-blue-500 group-hover:shadow-xl group-hover:scale-[1.02]"
										: "opacity-50"
								} transition-all duration-300 flex flex-col items-center justify-center p-6`}
							>
								<div
									className={`w-24 h-24 rounded-full bg-linear-to-br ${item.color} flex items-center justify-center text-white mb-4 shadow-lg`}
								>
									<Icon className="w-12 h-12" />
								</div>

								<h2 className="text-2xl font-bold text-gray-900 dark:text-white text-center">
									{item.label}
								</h2>

								{!item.path && (
									<p className="text-sm text-gray-500 dark:text-gray-400 mt-2">
										Coming soon
									</p>
								)}
							</div>
						</div>
					);
				})}

				{/* Empty slots */}
				{Array.from({ length: emptySlots }, (_, i) => (
					<div
						key={`empty-${i}`}
						className="relative flex flex-col items-center justify-center border border-gray-200 dark:border-gray-800"
					>
						<div className="absolute inset-0 m-5 bg-gray-50 dark:bg-gray-900/50 rounded-2xl border-4 border-dashed border-gray-300 dark:border-gray-700 flex items-center justify-center">
							<p className="text-gray-400 dark:text-gray-600 text-sm"></p>
						</div>
					</div>
				))}
			</div>
		</div>
	);
}
