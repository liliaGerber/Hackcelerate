import TeamMemberCard from "@/components/TeamMemberCard";
import teamMembersJson from "../settings/teamMembers.json";
import {Grid2} from "@mui/material"; // Import JSON

const TeamScreen = () => {
    return (
        <div className="w-full min-h-screen flex justify-center items-center" style={{minHeight: "90vh", minWidth: "100vw"}}>
            <div className="content-center" style={{width: "90vw"}}>
                <Grid2
                    container
                    spacing={4}
                    justifyContent="center"
                    className="w-full flex justify-center"
                >
                    {teamMembersJson.map((member) => (
                        <Grid2
                            key={member.name}
                            xs={12} sm={6} md={4} lg={3}
                            className="flex justify-center"
                        >
                            <TeamMemberCard teamMember={member} />
                        </Grid2>
                    ))}
                </Grid2>
            </div>
        </div>
    );
};

export default TeamScreen;
