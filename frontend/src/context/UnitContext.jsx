import React, { createContext, useContext, useState } from 'react';

const UnitContext = createContext(null);

export const UnitProvider = ({ children }) => {
  const [areaUnit, setAreaUnit] = useState('acres'); // 'acres' | 'hectares'
  const [volumeUnit, setVolumeUnit] = useState('liters'); // 'liters' | 'm3'

  const toggleAreaUnit = () => {
    setAreaUnit(prev => prev === 'acres' ? 'hectares' : 'acres');
  };

  const toggleVolumeUnit = () => {
    setVolumeUnit(prev => prev === 'liters' ? 'm3' : 'liters');
  };

  const formatVolume = (liters) => {
    if (liters === undefined || liters === null) return '0 L';
    if (volumeUnit === 'm3') {
      const m3 = (liters / 1000).toLocaleString(undefined, { maximumFractionDigits: 2 });
      return `${m3} m³`;
    }
    return `${liters.toLocaleString(undefined, { maximumFractionDigits: 0 })} L`;
  };

  const formatCurrency = (val) => {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      maximumFractionDigits: 0
    }).format(val || 0);
  };

  return (
    <UnitContext.Provider value={{
      areaUnit,
      setAreaUnit,
      volumeUnit,
      setVolumeUnit,
      toggleAreaUnit,
      toggleVolumeUnit,
      formatVolume,
      formatCurrency
    }}>
      {children}
    </UnitContext.Provider>
  );
};

export const useUnit = () => useContext(UnitContext);
