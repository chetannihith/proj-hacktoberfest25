# Random Person Generator

A dynamic React application that generates random user profiles with detailed information, built with modern web technologies and a focus on user experience.
## Features

### Core Functionality
- Generate random user profiles with realistic data
- Bulk generation of multiple profiles
- Individual and bulk regeneration options
- Export generated data to JSON

### Profile Information
- Full names with initials
- Age and work experience
- Occupation and salary
- Location
- Personality traits
- Hobbies and interests

### Interactive Features
- Real-time search functionality
- Advanced filtering system:
  - Age range filter
  - Salary range filter
- Animated transitions and loading states
- Responsive design for all screen sizes

### UI Components
- Color-coded profile cards
- Interactive badges
- Gradient backgrounds
- Hover effects and animations
- Loading indicators
- Empty state handlers

## Technologies Used

- **React** - Frontend framework
- **Tailwind CSS** - Styling and responsive design
- **Lucide React** - Icon components
- **shadcn/ui** - UI component library

## Installation

1. Clone the repository:
```bash
cd random-person-generator
```

2. Install dependencies:
```bash
npm install
# or
yarn install
```

3. Install required shadcn/ui components:
```bash
npx shadcn-ui@latest add card
npx shadcn-ui@latest add button
npx shadcn-ui@latest add badge
npx shadcn-ui@latest add input
npx shadcn-ui@latest add slider
```

4. Start the development server:
```bash
npm run dev
# or
yarn dev
```

## Usage

### Basic Operations

1. **Generate New Profile**
   - Click the "Generate New" button to create a single profile
   - Each profile includes random but realistic data

2. **Regenerate All**
   - Use the "Regenerate All" button to refresh all existing profiles
   - Maintains the same number of profiles but with new data

3. **Clear All**
   - Remove all generated profiles with the "Clear All" button

### Search and Filters

1. **Search**
   - Use the search bar to filter profiles by:
     - Name
     - Occupation
     - Location

2. **Advanced Filters**
   - Click the "Filters" button to access:
     - Age range selector
     - Salary range selector

### Export Data

1. Click the "Export" button to download all generated profiles
2. Data is exported in JSON format
3. Exported file is named `generated_people.json`


## Data Generation

The application generates random but realistic data using predefined sets of:
- First names
- Last names
- Occupations
- Locations
- Hobbies
- Personality traits

### Data Constraints

- Age: 20-65 years
- Salary: $30,000-$150,000
- Experience: Calculated based on age
- Hobbies: 2 unique hobbies per profile

## Customization

### Adding New Data Sets

To add new options for generated data, modify the arrays at the top of the component:
```javascript
const firstNames = [...];
const lastNames = [...];
const occupations = [...];
// etc.
```

### Styling

The application uses Tailwind CSS classes for styling. Major style elements include:
- Color schemes for profile cards
- Gradient backgrounds
- Animation classes
- Responsive design utilities

## Performance Considerations

- Debounced search functionality
- Optimized filtering logic
- Lazy loading for large datasets
- Smooth animations without performance impact

## Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Acknowledgments

- [shadcn/ui](https://ui.shadcn.com/) for the beautiful UI components
- [Lucide](https://lucide.dev/) for the icon set
- [Tailwind CSS](https://tailwindcss.com/) for the utility-first CSS framework