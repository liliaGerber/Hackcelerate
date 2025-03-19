import { Card, CardContent, Typography, CardActions, Button } from "@mui/material";
import LinkedInIcon from "@mui/icons-material/LinkedIn";
import EmailIcon from "@mui/icons-material/Email";
import PhoneIcon from "@mui/icons-material/Phone";

const TeamMemberCard = ({ teamMember }) => {
    return (
        <Card className="w-full max-w-[380px] lg:max-w-[400px] rounded-lg shadow-lg hover:shadow-xl transition transform hover:-translate-y-2 bg-white flex flex-col items-center text-center min-h-[450px] p-6">
            {/* Profile Image */}
            <div className="w-32 h-32 md:w-36 md:h-36 rounded-full overflow-hidden border-4 border-white shadow-md flex justify-center items-center">
                <img
                    src={teamMember.profilePic}
                    alt={teamMember.name}
                    className="w-full h-full object-cover"
                />
            </div>

            <CardContent className="flex flex-col items-center flex-grow">
                <Typography variant="h6" className="font-bold text-xl">
                    {teamMember.name}
                </Typography>
                <Typography variant="subtitle1" className="text-lg text-gray-500">
                    {teamMember.role}
                </Typography>
                <Typography variant="body2" className="mt-2 text-gray-700 leading-relaxed">
                    {teamMember.description}
                </Typography>
            </CardContent>

            {/* Social Buttons - Increased Size & Reduced Spacing */}
            <CardActions className="flex justify-center gap-1 pt-3 pb-5 w-full">
                <Button href={teamMember.linkedin} target="_blank" className="w-12 h-12 rounded-full bg-blue-600 text-white hover:bg-blue-700 flex justify-center items-center">
                    <LinkedInIcon fontSize="large" />
                </Button>
                <Button href={`mailto:${teamMember.email}`} className="w-12 h-12 rounded-full bg-red-500 text-white hover:bg-red-600 flex justify-center items-center">
                    <EmailIcon fontSize="large" />
                </Button>
                <Button href={`tel:${teamMember.phone}`} className="w-12 h-12 rounded-full bg-green-500 text-white hover:bg-green-600 flex justify-center items-center">
                    <PhoneIcon fontSize="large" />
                </Button>
            </CardActions>
        </Card>
    );
};

export default TeamMemberCard;
