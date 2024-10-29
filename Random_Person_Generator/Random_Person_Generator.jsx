import React, { useState, useEffect } from 'react';
import { RefreshCw, Search, SlidersHorizontal, Download } from 'lucide-react';
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Slider } from '@/components/ui/slider';

const firstNames = [
  "Emma", "Liam", "Olivia", "Noah", "Ava", "Ethan", "Sophia", "Mason",
  "Isabella", "William", "Mia", "James", "Charlotte", "Alexander", "Amelia",
  "Michael", "Harper", "Benjamin", "Evelyn", "Daniel"
];

const lastNames = [
  "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
  "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
  "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin"
];

const occupations = [
  "Software Engineer", "Teacher", "Doctor", "Artist", "Chef",
  "Architect", "Writer", "Scientist", "Designer", "Entrepreneur",
  "Lawyer", "Photographer", "Musician", "Nurse", "Marketing Manager"
];

const locations = [
  "New York, USA", "London, UK", "Tokyo, Japan", "Paris, France",
  "Sydney, Australia", "Toronto, Canada", "Berlin, Germany",
  "Singapore", "Dubai, UAE", "Amsterdam, Netherlands"
];

const hobbies = [
  "Photography", "Hiking", "Gaming", "Cooking", "Reading",
  "Painting", "Travel", "Music", "Yoga", "Gardening",
  "Swimming", "Chess", "Dancing", "Writing", "Cycling"
];

const personalities = [
  "Outgoing", "Creative", "Analytical", "Adventurous", "Empathetic",
  "Organized", "Energetic", "Thoughtful", "Confident", "Curious"
];

const NameGenerator = () => {
  const [people, setPeople] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [filters, setFilters] = useState({
    showFilters: false,
    minAge: 20,
    maxAge: 65,
    minSalary: 30000,
    maxSalary: 150000
  });
  const [loading, setLoading] = useState(false);

  const getRandomElement = (array) => array[Math.floor(Math.random() * array.length)];
  const getRandomAge = () => Math.floor(Math.random() * (filters.maxAge - filters.minAge) + filters.minAge);
  const getRandomSalary = () => Math.floor(Math.random() * (filters.maxSalary - filters.minSalary) + filters.minSalary);
  const getRandomExperience = (age) => Math.min(Math.floor((age - 20) * 0.8), Math.floor(Math.random() * 20));

  const getRandomColor = () => {
    const colors = [
      'bg-blue-100 text-blue-700',
      'bg-green-100 text-green-700',
      'bg-purple-100 text-purple-700',
      'bg-pink-100 text-pink-700',
      'bg-yellow-100 text-yellow-700',
      'bg-indigo-100 text-indigo-700',
      'bg-orange-100 text-orange-700',
      'bg-teal-100 text-teal-700'
    ];
    return getRandomElement(colors);
  };

  const generateNewPerson = () => {
    const age = getRandomAge();
    const firstName = getRandomElement(firstNames);
    const lastName = getRandomElement(lastNames);
    
    return {
      id: Date.now() + Math.random(),
      name: `${firstName} ${lastName}`,
      initials: `${firstName[0]}${lastName[0]}`,
      age: age,
      occupation: getRandomElement(occupations),
      location: getRandomElement(locations),
      hobbies: Array.from(new Set([getRandomElement(hobbies), getRandomElement(hobbies)])),
      personality: getRandomElement(personalities),
      experience: getRandomExperience(age),
      salary: getRandomSalary(),
      colorClass: getRandomColor(),
    };
  };

  const addPerson = async () => {
    setLoading(true);
    // Simulate network delay for more realistic feeling
    await new Promise(resolve => setTimeout(resolve, 500));
    setPeople(prev => [...prev, generateNewPerson()]);
    setLoading(false);
  };

  const regenerateAll = async () => {
    setLoading(true);
    await new Promise(resolve => setTimeout(resolve, 800));
    setPeople(prev => prev.map(() => generateNewPerson()));
    setLoading(false);
  };

  const clearAll = () => {
    setPeople([]);
  };

  const formatSalary = (salary) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      maximumFractionDigits: 0
    }).format(salary);
  };

  const exportData = () => {
    const dataStr = JSON.stringify(people, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    const exportFileDefaultName = 'generated_people.json';

    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
  };

  const filteredPeople = people.filter(person => {
    const matchesSearch = person.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         person.occupation.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         person.location.toLowerCase().includes(searchTerm.toLowerCase());
    
    const matchesFilters = person.age >= filters.minAge &&
                          person.age <= filters.maxAge &&
                          person.salary >= filters.minSalary &&
                          person.salary <= filters.maxSalary;
    
    return matchesSearch && matchesFilters;
  });

  return (
    <div className="p-4 max-w-6xl mx-auto min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800">
      <div className="mb-6 space-y-4">
        <div className="flex justify-between items-center">
          <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-purple-600 to-pink-600">
            Random Person Generator
          </h1>
          <div className="space-x-2">
            <Button onClick={addPerson} disabled={loading}>
              {loading ? "Generating..." : "Generate New"}
            </Button>
            <Button variant="outline" onClick={regenerateAll} disabled={loading || people.length === 0}>
              <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
              Regenerate All
            </Button>
            <Button variant="destructive" onClick={clearAll} disabled={people.length === 0}>
              Clear All
            </Button>
            <Button variant="outline" onClick={exportData} disabled={people.length === 0}>
              <Download className="w-4 h-4 mr-2" />
              Export
            </Button>
          </div>
        </div>
        
        <div className="flex gap-4">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
            <Input
              className="pl-10"
              placeholder="Search by name, occupation, or location..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>
          <Button
            variant="outline"
            onClick={() => setFilters(f => ({ ...f, showFilters: !f.showFilters }))}
          >
            <SlidersHorizontal className="w-4 h-4 mr-2" />
            Filters
          </Button>
        </div>

        {filters.showFilters && (
          <div className="p-4 bg-white dark:bg-gray-800 rounded-lg shadow-md space-y-4 animate-in slide-in-from-top duration-200">
            <div>
              <label className="text-sm font-medium">Age Range: {filters.minAge} - {filters.maxAge}</label>
              <Slider
                defaultValue={[filters.minAge, filters.maxAge]}
                max={80}
                min={18}
                step={1}
                onValueChange={([min, max]) => 
                  setFilters(f => ({ ...f, minAge: min, maxAge: max }))
                }
                className="my-2"
              />
            </div>
            <div>
              <label className="text-sm font-medium">
                Salary Range: {formatSalary(filters.minSalary)} - {formatSalary(filters.maxSalary)}
              </label>
              <Slider
                defaultValue={[filters.minSalary, filters.maxSalary]}
                max={200000}
                min={20000}
                step={5000}
                onValueChange={([min, max]) => 
                  setFilters(f => ({ ...f, minSalary: min, maxSalary: max }))
                }
                className="my-2"
              />
            </div>
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {filteredPeople.map((person) => (
          <Card 
            key={person.id} 
            className="overflow-hidden hover:shadow-lg transition-shadow duration-300 animate-in slide-in-from-bottom-4 duration-500"
          >
            <CardHeader className="flex flex-row items-center gap-4">
              <div className={`w-12 h-12 rounded-full flex items-center justify-center text-lg font-semibold ${person.colorClass} transition-colors duration-300`}>
                {person.initials}
              </div>
              <div>
                <CardTitle className="text-lg">{person.name}</CardTitle>
                <p className="text-sm text-gray-500">{person.age} years old</p>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="font-medium">Occupation:</span>
                  <span>{person.occupation}</span>
                </div>
                <div className="flex justify-between">
                  <span className="font-medium">Location:</span>
                  <span>{person.location}</span>
                </div>
                <div className="flex justify-between">
                  <span className="font-medium">Experience:</span>
                  <span>{person.experience} years</span>
                </div>
                <div className="flex justify-between">
                  <span className="font-medium">Salary:</span>
                  <span>{formatSalary(person.salary)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="font-medium">Personality:</span>
                  <span>{person.personality}</span>
                </div>
              </div>
              <div>
                <p className="font-medium mb-2">Hobbies:</p>
                <div className="flex flex-wrap gap-2">
                  {person.hobbies.map((hobby, index) => (
                    <Badge 
                      key={index} 
                      variant="secondary"
                      className="transition-all hover:scale-105"
                    >
                      {hobby}
                    </Badge>
                  ))}
                </div>
              </div>
            </CardContent>
            <CardFooter className="justify-end">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => {
                  setPeople(prev =>
                    prev.map(p =>
                      p.id === person.id ? generateNewPerson() : p
                    )
                  );
                }}
                className="hover:scale-105 transition-transform"
              >
                <RefreshCw className="w-4 h-4 mr-2" />
                Regenerate
              </Button>
            </CardFooter>
          </Card>
        ))}
      </div>
      
      {filteredPeople.length === 0 && people.length > 0 && (
        <div className="text-center py-10">
          <p className="text-gray-500">No results found for your search criteria.</p>
        </div>
      )}
    </div>
  );
};

export default NameGenerator;