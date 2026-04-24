import React from 'react';
import { Shield, ShieldAlert, ShieldCheck } from 'lucide-react';

const ConfidenceBadge = ({ confidence, retries }) => {
  const pct = Math.round(confidence * 100);

  const getConfig = () => {
    if (confidence >= 0.75) return {
      color: 'text-green-400',
      bg: 'bg-green-400/10 border-green-400/20',
      icon: <ShieldCheck size={14} />,
      label: 'Verified',
    };
    if (confidence >= 0.35) return {
      color: 'text-yellow-400',
      bg: 'bg-yellow-400/10 border-yellow-400/20',
      icon: <Shield size={14} />,
      label: 'Partial',
    };
    return {
      color: 'text-red-400',
      bg: 'bg-red-400/10 border-red-400/20',
      icon: <ShieldAlert size={14} />,
      label: 'Low',
    };
  };

  const { color, bg, icon, label } = getConfig();

  return (
    <div className="flex items-center gap-2 mt-2">
      <div className={`flex items-center gap-1 px-2 py-0.5 rounded-full border text-xs font-medium ${color} ${bg}`}>
        {icon}
        <span>{label} {pct}%</span>
      </div>
      {retries > 0 && (
        <span className="text-xs text-gray-500">
          {retries} retr{retries === 1 ? 'y' : 'ies'}
        </span>
      )}
    </div>
  );
};

export default ConfidenceBadge;