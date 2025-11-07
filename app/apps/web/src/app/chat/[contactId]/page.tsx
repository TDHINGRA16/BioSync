"use client";

import { useState, useEffect, useRef, use } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { ArrowLeft, Send, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useSocket } from "@/components/socket-provider";
import { useMessages, useInvalidateMessages, useMe } from "@/lib/hooks";
import { useUser } from "@clerk/nextjs";

interface Message {
  id: string;
  senderId: string;
  content: string;
  timestamp: string;
  isRead: boolean;
}

export default function ChatPage({ params }: { params: Promise<{ contactId: string }> }) {
  const { contactId } = use(params);
  const router = useRouter();
  const searchParams = useSearchParams();
  const isEyeTracking = searchParams.get("mode") === "eye-tracking";
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const { user } = useUser();
  const { socket, isConnected, sendMessage, startTyping, stopTyping, onReceiveMessage, onMessageDelivered, onUserTyping, onMessageSent } = useSocket();
  const invalidateMessages = useInvalidateMessages();

  // Fetch current user's profile to check if they are paralyzed
  const { data: currentUserData } = useMe();
  const isCurrentUserParalyzed = Boolean((currentUserData as any)?.isParalyzed);

  // Fetch messages from API (now includes otherUser data)
  const { data: messagesData, isLoading, error } = useMessages(contactId);
  const [messages, setMessages] = useState<Message[]>([]);

  const [inputMessage, setInputMessage] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [aiOptions, setAiOptions] = useState([
    "Sure! That sounds great! What time works for you?",
    "I'd love to, but I'm busy today. How about tomorrow?",
    "Thanks for asking! Let me check my schedule and get back to you.",
    "I'm not sure yet. Can I let you know in a bit?",
  ]);
  const [selectedOption, setSelectedOption] = useState<number | null>(null);
  const [dwellProgress, setDwellProgress] = useState<{ [key: number]: number }>({});

  // Update local messages when data is fetched
  useEffect(() => {
    if (messagesData?.messages) {
      setMessages(
        messagesData.messages.map((msg: any) => ({
          id: msg.id.toString(),
          senderId: msg.fromClerkId,
          content: msg.content,
          timestamp: new Date(msg.createdAt).toLocaleTimeString("en-US", {
            hour: "2-digit",
            minute: "2-digit",
          }),
          isRead: msg.isRead,
        }))
      );
    }
  }, [messagesData]);

  // Socket.IO event listeners
  useEffect(() => {
    if (!isConnected) return;

    // Listen for incoming messages (from others)
    const handleReceiveMessage = (data: any) => {
      if (data.fromClerkId === contactId || data.toClerkId === contactId) {
        const newMessage: Message = {
          id: data.messageId.toString(),
          senderId: data.fromClerkId,
          content: data.content,
          timestamp: new Date().toLocaleTimeString("en-US", {
            hour: "2-digit",
            minute: "2-digit",
          }),
          isRead: false,
        };
        
        // Add message if it doesn't already exist
        setMessages((prev) => {
          const exists = prev.some(m => m.id === newMessage.id);
          if (exists) return prev;
          return [...prev, newMessage];
        });
        
        // Mark as read if it's from the other person
        if (data.fromClerkId === contactId && data.messageId) {
          socket.markAsRead([data.messageId]);
        }
      }
    };

    // Listen for message sent confirmation (replace temp ID with real ID)
    const handleMessageSent = (data: any) => {
      setMessages((prev) => {
        // Find the most recent temp message and replace it
        const tempIndex = prev.findIndex(m => m.id.toString().startsWith('temp-'));
        if (tempIndex !== -1) {
          const updatedMessages = [...prev];
          updatedMessages[tempIndex] = {
            ...updatedMessages[tempIndex],
            id: data.messageId.toString(),
          };
          return updatedMessages;
        }
        return prev;
      });
    };

    // Listen for message delivered status
    const handleMessageDelivered = (data: any) => {
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === data.messageId.toString() ? { ...msg, isRead: data.status === "read" } : msg
        )
      );
    };

    // Listen for typing indicator
    const handleUserTyping = (data: any) => {
      if (data.fromClerkId === contactId) {
        setIsTyping(data.isTyping);
      }
    };

    onMessageSent(handleMessageSent);
    onReceiveMessage(handleReceiveMessage);
    onMessageDelivered(handleMessageDelivered);
    onUserTyping(handleUserTyping);

    return () => {
      // Cleanup is handled in SocketIOClient class
    };
  }, [isConnected, contactId, socket, onReceiveMessage, onMessageDelivered, onUserTyping, onMessageSent]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const handleSendMessage = (content: string) => {
    if (!isConnected || !user?.id) return;

    // Send message via Socket.IO using context method
    sendMessage(contactId, content);

    // Optimistically add message to UI
    const tempId = `temp-${Date.now()}`;
    const tempMessage: Message = {
      id: tempId,
      senderId: user.id,
      content,
      timestamp: new Date().toLocaleTimeString("en-US", {
        hour: "2-digit",
        minute: "2-digit",
      }),
      isRead: false,
    };
    setMessages((prev) => [...prev, tempMessage]);
    setInputMessage("");

    // Simulate AI generating new options after sending
    if (isEyeTracking) {
      setTimeout(() => {
        setAiOptions([
          "That's interesting! Tell me more about it.",
          "I completely agree with you on that.",
          "Hmm, I need to think about that for a moment.",
          "Could you explain that a bit more?",
        ]);
      }, 1000);
    }
  };

  const handleInputChange = (value: string) => {
    setInputMessage(value);
    
    // Send typing indicator using context method
    if (value.length > 0) {
      startTyping(contactId);
    } else {
      stopTyping(contactId);
    }
  };

  const handleOptionHover = (index: number, isHovering: boolean) => {
    if (!isEyeTracking) return;

    if (isHovering) {
      // Simulate dwell timer
      let progress = 0;
      const interval = setInterval(() => {
        progress += 10;
        setDwellProgress((prev) => ({ ...prev, [index]: progress }));

        if (progress >= 100) {
          clearInterval(interval);
          handleSendMessage(aiOptions[index]);
          setSelectedOption(index);
          setTimeout(() => {
            setSelectedOption(null);
            setDwellProgress({});
          }, 500);
        }
      }, 100);

      return () => clearInterval(interval);
    } else {
      setDwellProgress((prev) => ({ ...prev, [index]: 0 }));
    }
  };

  if (isLoading) {
    return (
      <div className="h-screen flex items-center justify-center bg-white dark:bg-gray-950">
        <div className="text-center">
          <Loader2 className="w-12 h-12 animate-spin mx-auto mb-4 text-blue-500" />
          <p className="text-gray-600 dark:text-gray-400">Loading conversation...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="h-screen flex items-center justify-center bg-white dark:bg-gray-950">
        <div className="text-center">
          <p className="text-red-500 mb-4">Failed to load conversation</p>
          <p className="text-gray-600 dark:text-gray-400 text-sm">
            {error instanceof Error ? error.message : "Unknown error"}
          </p>
          <Button onClick={() => router.push("/chat")} className="mt-4">
            Back to Contacts
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen flex flex-col bg-white dark:bg-gray-950">
      {/* Top Bar */}
      <div className="h-16 bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-800 flex items-center px-4 gap-4">
        <Button
          variant="ghost"
          size="icon"
          onClick={() => router.push("/chat")}
          className="w-10 h-10"
        >
          <ArrowLeft className="w-6 h-6" />
        </Button>

        <div className="flex items-center gap-3 flex-1">
          <div className="relative">
            <div className="w-12 h-12 rounded-full bg-linear-to-br from-blue-500 to-purple-600 flex items-center justify-center text-white text-lg font-bold">
              {messagesData?.otherUser?.nickname?.slice(0, 2).toUpperCase() || contactId.slice(0, 2).toUpperCase()}
            </div>
            <div
              className={`absolute bottom-0 right-0 w-4 h-4 rounded-full border-2 border-white dark:border-gray-900 ${
                isConnected ? "bg-green-500" : "bg-gray-400"
              }`}
            />
          </div>
          <div>
            <h2 className="text-lg font-bold text-gray-900 dark:text-white">
              {messagesData?.otherUser?.nickname || contactId}
            </h2>
            <p className="text-sm text-gray-500">
              {isTyping ? "Typing..." : isConnected ? "Online" : "Offline"}
            </p>
          </div>
        </div>

        {isEyeTracking && (
          <div className="px-4 py-2 bg-purple-100 dark:bg-purple-900 text-purple-700 dark:text-purple-300 rounded-lg text-sm font-semibold">
            Eye-Tracking Mode
          </div>
        )}
      </div>

      {/* Split Layout for Paralyzed Users (current user is paralyzed) */}
      {isCurrentUserParalyzed && !isEyeTracking ? (
        <div className="flex flex-1 overflow-hidden">
          {/* Left Side - Messages */}
          <div className="flex flex-col w-1/2 min-w-[50vw] max-w-[50vw] border-r border-gray-200 dark:border-gray-800">
            <div className="flex-1 overflow-y-auto p-6 space-y-4 bg-white dark:bg-gray-950">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex ${
                    message.senderId === user?.id ? "justify-end" : "justify-start"
                  }`}
                >
                  <div
                    className={`flex gap-2 max-w-[70%] ${
                      message.senderId === user?.id ? "flex-row-reverse" : ""
                    }`}
                  >
                    {message.senderId !== user?.id && (
                      <div className="w-10 h-10 rounded-full bg-linear-to-br from-blue-500 to-purple-600 flex items-center justify-center text-white text-sm font-bold shrink-0">
                        {messagesData?.otherUser?.nickname?.slice(0, 2).toUpperCase() || contactId.slice(0, 2).toUpperCase()}
                      </div>
                    )}
                    <div>
                      <div
                        className={`rounded-xl px-4 py-3 shadow-sm ${
                          message.senderId === user?.id
                            ? "bg-blue-500 text-white"
                            : "bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-white"
                        }`}
                      >
                        <p className="text-base leading-relaxed">{message.content}</p>
                      </div>
                      <p className="text-xs text-gray-400 mt-1 px-2 flex items-center gap-1">
                        {message.timestamp}
                        {message.senderId === user?.id && message.isRead && (
                          <span className="text-blue-500">✓✓</span>
                        )}
                      </p>
                    </div>
                  </div>
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>

            <div className="h-24 bg-white dark:bg-gray-900 border-t border-gray-200 dark:border-gray-800 flex items-center px-6 gap-4">
              <Input
                value={inputMessage}
                onChange={(e) => handleInputChange(e.target.value)}
                onKeyPress={(e) => {
                  if (e.key === "Enter" && inputMessage.trim()) {
                    handleSendMessage(inputMessage);
                    stopTyping(contactId);
                  }
                }}
                placeholder="Type a message..."
                className="flex-1 h-14 text-base"
                disabled={!isConnected}
              />
              <Button
                onClick={() => {
                  if (inputMessage.trim()) {
                    handleSendMessage(inputMessage);
                    stopTyping(contactId);
                  }
                }}
                disabled={!inputMessage.trim() || !isConnected}
                size="icon"
                className="w-14 h-14 rounded-full bg-blue-500 hover:bg-blue-600"
              >
                <Send className="w-5 h-5" />
              </Button>
            </div>
          </div>

          {/* Right Side - AI Response Button */}
          <div className="w-1/2 min-w-[50vw] max-w-[50vw] border-l border-gray-200 dark:border-gray-800 bg-gray-50 dark:bg-gray-900 flex">
            <Button
              onClick={() => router.push(`/chat/${contactId}/airesponses`)}
              className="w-full h-full rounded-none bg-linear-to-br from-purple-500 to-blue-500 hover:opacity-90 text-white text-3xl font-bold tracking-wide"
            >
              AI Response Options
            </Button>
          </div>
        </div>
      ) : (
        <>
          {/* Messages Area */}
          <div
            className={`flex-1 overflow-y-auto p-4 space-y-3 ${
              isEyeTracking ? "max-h-[40vh]" : ""
            }`}
          >
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex ${
                  message.senderId === user?.id ? "justify-end" : "justify-start"
                }`}
              >
                <div
                  className={`flex gap-2 max-w-[70%] ${
                    message.senderId === user?.id ? "flex-row-reverse" : ""
                  }`}
                >
                  {message.senderId !== user?.id && (
                    <div className="w-10 h-10 rounded-full bg-linear-to-br from-blue-500 to-purple-600 flex items-center justify-center text-white text-sm font-bold shrink-0">
                      {messagesData?.otherUser?.nickname?.slice(0, 2).toUpperCase() || contactId.slice(0, 2).toUpperCase()}
                    </div>
                  )}
                  <div>
                    <div
                      className={`rounded-xl px-4 py-3 shadow-sm ${
                        message.senderId === user?.id
                          ? "bg-blue-500 text-white"
                          : "bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-white"
                      }`}
                    >
                      <p className="text-base leading-relaxed">{message.content}</p>
                    </div>
                    <p className="text-xs text-gray-400 mt-1 px-2 flex items-center gap-1">
                      {message.timestamp}
                      {message.senderId === user?.id && message.isRead && (
                        <span className="text-blue-500">✓✓</span>
                      )}
                    </p>
                  </div>
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>

          {/* Input Area or AI Options */}
          {!isEyeTracking ? (
            // Regular User Input
            <div className="h-20 bg-white dark:bg-gray-900 border-t border-gray-200 dark:border-gray-800 flex items-center px-4 gap-3">
              <Input
                value={inputMessage}
                onChange={(e) => handleInputChange(e.target.value)}
                onKeyPress={(e) => {
                  if (e.key === "Enter" && inputMessage.trim()) {
                    handleSendMessage(inputMessage);
                    stopTyping(contactId);
                  }
                }}
                placeholder="Type a message..."
                className="flex-1 h-12 text-base"
                disabled={!isConnected}
              />
              <Button
                onClick={() => {
                  if (inputMessage.trim()) {
                    handleSendMessage(inputMessage);
                    stopTyping(contactId);
                  }
                }}
                disabled={!inputMessage.trim() || !isConnected}
                size="icon"
                className="w-12 h-12 rounded-full bg-blue-500 hover:bg-blue-600"
              >
                <Send className="w-5 h-5" />
              </Button>
            </div>
          ) : (
            // Eye-Tracking AI Options
            <div className="flex-1 bg-gray-50 dark:bg-gray-900 p-6">
              <div className="grid grid-cols-2 gap-5 h-full">
                {aiOptions.slice(0, 2).map((option, index) => (
                  <button
                    key={index}
                    onMouseEnter={() => handleOptionHover(index, true)}
                    onMouseLeave={() => handleOptionHover(index, false)}
                    onClick={() => handleSendMessage(option)}
                    className={`relative min-h-[180px] rounded-2xl p-6 text-white text-2xl font-bold shadow-lg transition-all duration-300 hover:scale-[1.02] hover:shadow-2xl ${
                      index === 0
                        ? "bg-blue-500 hover:ring-4 hover:ring-blue-300"
                        : "bg-green-500 hover:ring-4 hover:ring-green-300"
                    } ${selectedOption === index ? "scale-105 ring-4 ring-white" : ""}`}
                  >
                    {/* Number Badge with Dwell Progress */}
                    <div className="absolute top-4 left-4 w-12 h-12 rounded-full bg-white bg-opacity-30 flex items-center justify-center text-3xl font-black">
                      {dwellProgress[index] > 0 && (
                        <svg
                          className="absolute inset-0 w-12 h-12 -rotate-90"
                          viewBox="0 0 48 48"
                        >
                          <circle
                            cx="24"
                            cy="24"
                            r="20"
                            stroke="white"
                            strokeWidth="4"
                            fill="none"
                            strokeDasharray={`${2 * Math.PI * 20}`}
                            strokeDashoffset={`${
                              2 * Math.PI * 20 * (1 - dwellProgress[index] / 100)
                            }`}
                            className="transition-all duration-100"
                          />
                        </svg>
                      )}
                      {index + 1}
                    </div>

                    {/* Checkmark Animation */}
                    {selectedOption === index && (
                      <div className="absolute inset-0 flex items-center justify-center">
                        <div className="w-20 h-20 rounded-full bg-white flex items-center justify-center animate-bounce">
                          <svg
                            className="w-12 h-12 text-green-500"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={3}
                              d="M5 13l4 4L19 7"
                            />
                          </svg>
                        </div>
                      </div>
                    )}

                    {/* Option Text */}
                    <div className="flex items-center justify-center h-full text-center px-4">
                      {option}
                    </div>
                  </button>
                ))}
                
                {/* Previous Button - First card of bottom row */}
                <button
                  onClick={() => router.push("/chat")}
                  className="relative min-h-[180px] rounded-2xl p-6 text-white text-2xl font-bold shadow-lg transition-all duration-300 hover:scale-[1.02] hover:shadow-2xl bg-gray-600 hover:ring-4 hover:ring-gray-400"
                >
                  <div className="flex flex-col items-center justify-center h-full gap-4">
                    <ArrowLeft className="w-16 h-16" />
                    <span>Previous</span>
                  </div>
                </button>

                {aiOptions.slice(2, 3).map((option, index) => (
                  <button
                    key={index + 2}
                    onMouseEnter={() => handleOptionHover(index + 2, true)}
                    onMouseLeave={() => handleOptionHover(index + 2, false)}
                    onClick={() => handleSendMessage(option)}
                    className="relative min-h-[180px] rounded-2xl p-6 text-white text-2xl font-bold shadow-lg transition-all duration-300 hover:scale-[1.02] hover:shadow-2xl bg-purple-500 hover:ring-4 hover:ring-purple-300"
                  >
                    {/* Number Badge with Dwell Progress */}
                    <div className="absolute top-4 left-4 w-12 h-12 rounded-full bg-white bg-opacity-30 flex items-center justify-center text-3xl font-black">
                      {dwellProgress[index + 2] > 0 && (
                        <svg
                          className="absolute inset-0 w-12 h-12 -rotate-90"
                          viewBox="0 0 48 48"
                        >
                          <circle
                            cx="24"
                            cy="24"
                            r="20"
                            stroke="white"
                            strokeWidth="4"
                            fill="none"
                            strokeDasharray={`${2 * Math.PI * 20}`}
                            strokeDashoffset={`${
                              2 * Math.PI * 20 * (1 - dwellProgress[index + 2] / 100)
                            }`}
                            className="transition-all duration-100"
                          />
                        </svg>
                      )}
                      {index + 3}
                    </div>

                    {/* Checkmark Animation */}
                    {selectedOption === index + 2 && (
                      <div className="absolute inset-0 flex items-center justify-center">
                        <div className="w-20 h-20 rounded-full bg-white flex items-center justify-center animate-bounce">
                          <svg
                            className="w-12 h-12 text-green-500"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={3}
                              d="M5 13l4 4L19 7"
                            />
                          </svg>
                        </div>
                      </div>
                    )}

                    {/* Option Text */}
                    <div className="flex items-center justify-center h-full text-center px-4">
                      {option}
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

